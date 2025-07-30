from model import build_transformer
from dataset import BilingualDataset, causal_mask
from config import get_config, get_weights_file_path, latest_weights_file_path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import LambdaLR

import warnings
from tqdm import tqdm
import os
from pathlib import Path
import unicodedata
import re

# Huggingface datasets and tokenizers
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

import torchmetrics
from torch.utils.tensorboard import SummaryWriter

def preprocess_tamil_text(text):
    """
    Enhanced Tamil text preprocessing with Unicode normalization.
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Critical: Unicode normalization for Tamil
    text = unicodedata.normalize('NFC', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove control characters
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
    
    return text

def quality_filter(example):
    """
    Filter function to ensure high-quality translation pairs.
    This is important for samanantar due to its mixed domain sources.
    """
    try:
        src_text = example['src'].strip()
        tgt_text = example['tgt'].strip()
        
        # Basic length checks
        if len(src_text) == 0 or len(tgt_text) == 0:
            return False
        
        # Word count checks
        src_words = len(src_text.split())
        tgt_words = len(tgt_text.split())
        
        # Filter very short or very long sentences
        if src_words < 3 or src_words > 100 or tgt_words < 3 or tgt_words > 100:
            return False
        
        # Length ratio check (Tamil tends to be longer than English)
        ratio = tgt_words / src_words if src_words > 0 else 0
        if ratio < 0.3 or ratio > 3.0:  # More permissive for Tamil
            return False
        
        # Character length checks
        if len(src_text) > 500 or len(tgt_text) > 800:  # Tamil can be longer
            return False
            
        return True
    except:
        return False

def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # Precompute the encoder output and reuse it for every step
    encoder_output = model.encode(source, source_mask)
    # Initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break

        # build mask for target
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # calculate output
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # get next token
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1
        )

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)

def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step, writer, num_examples=2):
    model.eval()
    count = 0

    source_texts = []
    expected = []
    predicted = []

    try:
        # get the console window width
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        # If we can't get the console width, use 80 as default
        console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device) # (b, seq_len)
            encoder_mask = batch["encoder_mask"].to(device) # (b, 1, 1, seq_len)

            # check that the batch size is 1
            assert encoder_input.size(
                0) == 1, "Batch size must be 1 for validation"

            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)
            
            # Print the source, target and model output
            print_msg('-'*console_width)
            print_msg(f"{f'SOURCE: ':>12}{source_text}")
            print_msg(f"{f'TARGET: ':>12}{target_text}")
            print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

            if count == num_examples:
                print_msg('-'*console_width)
                break
    
    if writer:
        # Evaluate the character error rate
        # Compute the char error rate 
        metric = torchmetrics.CharErrorRate()
        cer = metric(predicted, expected)
        writer.add_scalar('validation cer', cer, global_step)
        writer.flush()

        # Compute the word error rate
        metric = torchmetrics.WordErrorRate()
        wer = metric(predicted, expected)
        writer.add_scalar('validation wer', wer, global_step)
        writer.flush()

        # Compute the BLEU metric
        metric = torchmetrics.BLEUScore()
        bleu = metric(predicted, expected)
        writer.add_scalar('validation BLEU', bleu, global_step)
        writer.flush()

def get_all_sentences(ds, lang):
    """
    Updated to handle samanantar dataset structure.
    """
    for item in ds:
        if lang == "en":
            yield item['src']  # English source
        elif lang == "ta":
            # Preprocess Tamil text during tokenizer training
            tamil_text = preprocess_tamil_text(item['tgt'])
            yield tamil_text
        else:
            raise ValueError(f"Unsupported language: {lang}")

def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        print(f"Building tokenizer for {lang}...")
        # Most code taken from: https://huggingface.co/docs/tokenizers/quicktour
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        
        # Adjusted parameters for Tamil - larger vocabulary for morphologically rich language
        min_freq = 2 if lang == "en" else 1  # Lower frequency threshold for Tamil
        trainer = WordLevelTrainer(
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], 
            min_frequency=min_freq,
            vocab_size=8000 if lang == "ta" else 6000  # Larger vocab for Tamil
        )
        
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
        print(f"Tokenizer for {lang} saved to {tokenizer_path}")
    else:
        print(f"Loading existing tokenizer for {lang} from {tokenizer_path}")
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_ds(config):
    print(f"Loading dataset: {config['datasource']}")
    
    # Load the samanantar dataset with Tamil configuration
    try:
        ds_raw = load_dataset(config['datasource'], config['dataset_config'], split='train')
        print(f"Loaded dataset with {len(ds_raw)} examples")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Falling back to streaming mode...")
        ds_raw = load_dataset(config['datasource'], config['dataset_config'], split='train', streaming=True)
        # Convert to regular dataset if streaming
        ds_raw = ds_raw.take(100000)  # Take first 100k for memory management
        ds_raw = list(ds_raw)
    
    # Apply quality filtering if enabled
    if config.get('quality_filter', True):
        print("Applying quality filters...")
        initial_size = len(ds_raw)
        ds_raw = ds_raw.filter(quality_filter)
        filtered_size = len(ds_raw)
        print(f"Filtered dataset: {initial_size} -> {filtered_size} examples")
    
    # Limit dataset size if specified (for testing/development)
    if config.get('max_train_samples') and config['max_train_samples'] > 0:
        max_samples = min(config['max_train_samples'], len(ds_raw))
        ds_raw = ds_raw.select(range(max_samples))
        print(f"Limited dataset to {max_samples} samples")

    # Build tokenizers
    print("Building/loading tokenizers...")
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

    print(f"Source tokenizer vocab size: {tokenizer_src.get_vocab_size()}")
    print(f"Target tokenizer vocab size: {tokenizer_tgt.get_vocab_size()}")

    # Create train/validation split (samanantar only has train split)
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    print(f"Train size: {train_ds_size}, Validation size: {val_ds_size}")

    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])

    # Find the maximum length of each sentence in the source and target sentence
    max_len_src = 0
    max_len_tgt = 0

    print("Computing maximum sequence lengths...")
    sample_size = min(1000, len(ds_raw))  # Sample for length calculation
    for i in range(sample_size):
        item = ds_raw[i]
        src_ids = tokenizer_src.encode(item['src']).ids
        tgt_text = preprocess_tamil_text(item['tgt']) if config['lang_tgt'] == 'ta' else item['tgt']
        tgt_ids = tokenizer_tgt.encode(tgt_text).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_tgt}')
    
    if max_len_src > config['seq_len'] or max_len_tgt > config['seq_len']:
        print(f"WARNING: Some sequences are longer than seq_len ({config['seq_len']})")
        print(f"Consider increasing seq_len to at least {max(max_len_src, max_len_tgt) + 10}")

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config["seq_len"], config['seq_len'], d_model=config['d_model'])
    return model

def train_model(config):
    # Define the device
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"
    print("Using device:", device)
    if (device == 'cuda'):
        print(f"Device name: {torch.cuda.get_device_name(device.index)}")
        print(f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB")
    elif (device == 'mps'):
        print(f"Device name: <mps>")
    else:
        print("NOTE: If you have a GPU, consider using it for training.")
        print("      On a Windows machine with NVidia GPU, check this video: https://www.youtube.com/watch?v=GMSjDTU8Zlc")
        print("      On a Mac machine, run: pip3 install --pre torch torchvision torchaudio torchtext --index-url https://download.pytorch.org/whl/nightly/cpu")
    device = torch.device(device)

    # Make sure the weights folder exists
    model_folder = f"{config['datasource'].replace('/', '_')}_{config['model_folder']}"
    Path(model_folder).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Tensorboard
    writer = SummaryWriter(config['experiment_name'])

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    # Learning rate scheduler with warmup (helpful for Tamil training)
    def lr_lambda(step):
        warmup_steps = config.get('warmup_steps', 1000)
        if step < warmup_steps:
            return step / warmup_steps
        return 1.0

    scheduler = LambdaLR(optimizer, lr_lambda)

    # If the user specified a model to preload before training, load it
    initial_epoch = 0
    global_step = 0
    preload = config['preload']
    model_filename = latest_weights_file_path(config) if preload == 'latest' else get_weights_file_path(config, preload) if preload else None
    if model_filename:
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
        if 'scheduler_state_dict' in state:
            scheduler.load_state_dict(state['scheduler_state_dict'])
    else:
        print('No model to preload, starting from scratch')

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch, config['num_epochs']):
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
        epoch_loss = 0
        
        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device) # (b, seq_len)
            decoder_input = batch['decoder_input'].to(device) # (B, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # (B, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device) # (B, 1, seq_len, seq_len)

            # Run the tensors through the encoder, decoder and the projection layer
            encoder_output = model.encode(encoder_input, encoder_mask) # (B, seq_len, d_model)
            decoder_output = model.decode(decoder_input, encoder_output, encoder_mask, decoder_mask) # (B, seq_len, d_model)
            proj_output = model.project(decoder_output) # (B, seq_len, vocab_size)

            # Compare the output with the label
            label = batch['label'].to(device) # (B, seq_len)

            # Compute the loss using a simple cross entropy
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"})

            epoch_loss += loss.item()

            # Log the loss
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.add_scalar('learning rate', scheduler.get_last_lr()[0], global_step)
            writer.flush()

            # Backpropagate the loss
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Update the weights
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            
            # Update learning rate
            scheduler.step()

            global_step += 1

        avg_epoch_loss = epoch_loss / len(train_dataloader)
        print(f"Epoch {epoch:02d} - Average Loss: {avg_epoch_loss:.4f}")
        writer.add_scalar('epoch loss', avg_epoch_loss, epoch)

        # Run validation at the end of every epoch
        run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer)

        # Save the model at the end of every epoch
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'global_step': global_step
        }, model_filename)
        print(f"Model saved to {model_filename}")

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_config()
    print("Training configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    train_model(config)