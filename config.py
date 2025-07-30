from pathlib import Path

def get_config():
    return {
        "batch_size": 16,
        "num_epochs": 25,  # Increased for Tamil complexity
        "lr": 3e-5,  # More conservative learning rate for Tamil
        "seq_len": 400,  # Increased for Tamil morphological complexity
        "d_model": 512,
        "datasource": 'ai4bharat/samanantar',  # Changed dataset
        "lang_src": "en",
        "lang_tgt": "ta",  # Changed to Tamil
        "dataset_config": "ta",  # Dataset configuration for Tamil
        "model_folder": "weights",
        "model_basename": "tmodel_en_ta_",  # Updated model name
        "preload": "latest",
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel_en_ta",
        "warmup_steps": 1000,  # Added warmup for Tamil training
        "max_train_samples": None,  # Set to number to limit dataset size for testing
        "quality_filter": True,  # Enable quality filtering
    }

def get_weights_file_path(config, epoch: str):
    model_folder = f"{config['datasource'].replace('/', '_')}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)

# Find the latest weights file in the weights folder
def latest_weights_file_path(config):
    model_folder = f"{config['datasource'].replace('/', '_')}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])