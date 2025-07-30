# annotated version
# transformer - encoder - decoder structure
# encoder : input --> input embedding --> positional encoding --> encoder layer
# encoder layer is broken into --> multihead attention --> layer norm --> feed forward network --> layer norm
import torch
import torch.nn as nn
import math

class InputEmbedding(nn.Module):
	""" Input embedding - this is the first component of the transformer model
		it converts the input tokens into a dense vector representation of size d_model (512 usually)
	"""
	def __init__(self, d_model: int, vocab_size: int):
		"""
		Initializes the InputEmbedding module.
		Args:
			d_model (int): The dimension of the model (embedding size)
			vocab_size (int): The size of the vocabulary.
		"""
		super(InputEmbedding, self).__init__()
		self.vocab_size = vocab_size
		self.d_model = d_model
		self.embedding = nn.Embedding(vocab_size, d_model)

	def forward(self, x):
		"""
			self.embedding(x) returns the embedding of the input tensor x.
			According to the paper, the input embedding is multiplied by sqrt(d_model) to scale the embeddings.
		"""
		return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
	""" positional encoding - appended to the input embedding
		this is another 512 or d_model sized vector that is added to the input embedding
		it is used to give the model information about the position of the token in the sequence
		formula -  PE(pos, 2i) = sin(pos / 10000^(2i/d_model)),
		PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
		where pos is the position of the token in the sequence and i is the dimension of the embedding.
	"""
	def __init__(self, d_model: int = 512, seq_len: int = 1000, dropout: float = 0.1):
		"""
		Initializes the PositionalEncoding module.
		Args:
			d_model (int): The dimension of the model (embedding size).
			seq_len (int): The maximum length of the input sequences.
			dropout (float): Dropout rate.
		"""
		super(PositionalEncoding, self).__init__()
		self.d_model = d_model
		self.seq_len = seq_len
		self.dropout = nn.Dropout(dropout)  # FIXED: Added self. prefix
		#create a matrix of shape (seq_len, d_model)
		pe = torch.zeros(seq_len, d_model)
		#create a vector of shape (seq_len)
		position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
		#for the div term, we create a vector of shape (d_model)
		div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
		#calculate pe of even and odd indices by using the formula
		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)
		#add a batch dimension and register the pe as a buffer so that it is not a parameter of the model
		pe = pe.unsqueeze(0)
		self.register_buffer('pe', pe)

	def forward(self, x):
		"""
		Adds positional encoding to the input tensor x.
		Args:
			x (torch.Tensor): The input tensor of shape (batch_size, seq_len, d_model).
		Returns:
			torch.Tensor: The input tensor with positional encoding added.
		"""
		x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) # (batch, seq_len, d_model)
		return self.dropout(x)


class LayerNormalization(nn.Module):
	""" Layer normalization - this is used to normalize the input to the layer
		it is used to stabilize the training of the model.
		in the original transformers paper, there are 2 layer norm layers in the encoder layer -
			one after the multihead attention and one after the feed forward network.
	"""
	def __init__(self, d_model: int = 512, eps: float = 1e-6):
		"""
		Initializes the LayerNormalization module.
		Args:
			d_model (int): The dimension of the model (embedding size).
			eps (float): A small value to avoid division by zero.
		"""
		super(LayerNormalization, self).__init__()
		self.d_model = d_model
		self.eps = eps
		self.alpha = nn.Parameter(torch.ones(1))
		self.bias = nn.Parameter(torch.zeros(1))

	def forward(self, x):
		"""
		Applies layer normalization to the input tensor x.
		Args:
			x (torch.Tensor): The input tensor of shape (batch_size, seq_len, d_model).
		Returns:
			torch.Tensor: The normalized tensor.
		"""
		mean = x.mean(dim=-1, keepdim=True)
		variance = x.var(dim=-1, keepdim=True, unbiased=False)
		x_normalized = (x - mean) / torch.sqrt(variance + self.eps)
		return self.alpha * x_normalized + self.bias


class FeedForwardNetwork(nn.Module):
	""" in the paper, each of the layers in encoder and decoder contains a feed forward network
		it is a simple 2 layer fully connected network with ReLU activation in between.
		the first layer projects the input to a higher dimension (dff = 2048 usually) and the second layer projects it back to the original dimension (512 usually).
	"""
	def __init__(self, d_model: int = 512, d_ff: int = 2048, dropout: float = 0.1):
		"""
		Initializes the FeedForwardNetwork module.
		Args:
			d_model (int): The dimension of the model (embedding size).
			d_ff (int): The dimension of the feed forward network.
		"""
		super(FeedForwardNetwork, self).__init__()
		self.d_model = d_model
		self.d_ff = d_ff
		self.linear1 = nn.Linear(d_model, d_ff) # W1 and B1
		self.linear2 = nn.Linear(d_ff, d_model) #W2 and B2
		self.relu = nn.ReLU()
		self.dropout = nn.Dropout(dropout)  # dropout layer to prevent overfitting

	def forward(self, x):
		"""
		Applies the feed forward network to the input tensor x.
		(batch, seq_len, d_model) --> (batch, seq_len, d_ff) --> (batch, seq_len, d_model)
		Args:
			x (torch.Tensor): The input tensor of shape (batch_size, seq_len, d_model).
		Returns:
			torch.Tensor: The output tensor after applying the feed forward network.
		"""
		return self.linear2(self.dropout(self.relu(self.linear1(x))))


class MultiHeadAttention(nn.Module):
	"""
	single head self attention - self attention allows the model to relate words to each other.
	consider a simple case where we consider seq length as 6 and d_model or d_k as 512.
	attention(Q, K, V) = softmax(Q * K.T / sqrt(d_k)) * V
	The matrices Q, K and V are the input sentence.
	softmax(Q * K.T / sqrt(d_k)) in this case would be a matrix of shape (6, 6),
	and the values in the matrix is a dot product of embedding of each word to itself and the other words.
	Then we multiply this matrix with the value matrix V so we end up with a matrix of shape (6, 512).
	each row in the matrix captures not only the meaning(embedding) or the position, but also each word's interaction with the other words in the sentence.

	In multihead attention, the input is again split into Q, K and V matrices for the encoder.
	We multiply the input Q, K and V by 3 different weight matrices Wq, Wk and Wv to get Q', K' and V' matrices.
	the dimensions of Wq, Wk and Wv is (d_model, d_model) and the Q', K' and V' matrices has a shape of (seq_len, d_model).
	Wq, Wk and Wv are learned linear projections of the key, query and value matrices.

	We split these Q', K' and V' matrices into multiple heads [(Q1, Q2, Q3, Q4,..), (K1, K2, K3, K4,..), (V1, V2, V3, V4,..)]
	along the embedding dimension and not the sequence dimension. this means that each head will have access to the full sentence,
	but different part of the embedding of each word.
	We apply attention to the (Q1, K1, v1), (Q2, K2, V2), ... and so on in parallel and have attention scores for each of the heads.
	multihead(q, k, v) = concat(head1, head2, head3, ...) * W0
	where W0 is a weight matrix of shape (d_model, d_model) that projects the concatenated output of all the heads back to the original dimension (sequence length, d_model).
	"""
	def __init__(self, d_model: int = 512, num_heads: int = 8, dropout: float = 0.1):
		"""
		Initializes the MultiHeadAttention module.
		Args:
			d_model (int): The dimension of the model (embedding size).
			num_heads (int): The number of attention heads.
			dropout (float): Dropout rate.
		"""
		super(MultiHeadAttention, self).__init__()
		self.d_model = d_model
		self.num_heads = num_heads
		assert d_model % self.num_heads == 0, "d_model is not divisible by num heads"

		self.d_k = d_model // num_heads
		self.Wq = nn.Linear(d_model, d_model)  # weight matrix for query
		self.Wk = nn.Linear(d_model, d_model)  # weight matrix for key
		self.Wv = nn.Linear(d_model, d_model)  # weight matrix for value
		self.W0 = nn.Linear(d_model, d_model)  # weight matrix for output
		self.dropout = nn.Dropout(dropout)  # dropout layer to prevent overfitting

	@staticmethod
	def attention(query, key, value, mask, dropout: nn.Dropout):
		"""		Applies the attention mechanism to the input tensors.
		Args:
			query (torch.Tensor): The query tensor of shape (batch_size, num_heads, seq_len, d_k).
			key (torch.Tensor): The key tensor of shape (batch_size, num_heads, seq_len, d_k).
			value (torch.Tensor): The value tensor of shape (batch_size, num_heads, seq_len, d_k).
			mask (torch.Tensor): The mask tensor of shape (batch_size, 1, seq_len, seq_len) or None.
			dropout (nn.Dropout): The dropout layer to apply to the attention weights.
		"""
		d_k = query.shape[-1]
		#we apply the formula for attention - softmax(QK.T/sqrt(d_k))V
		#step 1 - we get the attention matrix which is (batch, h, seq_len, seq_len) shape from the input shape of (batch, h, seq_len, d_k)
		attention_scores = (query @ key.transpose(-2, -1))/math.sqrt(d_k)
		#if mask is applied, we want to set the attention scores to -inf for the masked positions. softmax will ignore these positions.
		if mask is not None:
			attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
		#step 2 - we apply softmax to the attention scores to get the attention weights
		attention_weights = torch.softmax(attention_scores, dim=-1)	#shape will be (batch, h, seq_len, seq_len)
		#step 3 - we apply dropout to the attention weights
		if dropout is not None:
			attention_weights = dropout(attention_weights)
		#step 4 - we multiply the attention weights with the value matrix to get the output
		output = attention_weights @ value  # shape will be (batch, h, seq_len, d_k)
		return output, attention_weights

	def forward(self, q, k, v, mask):
		""" multihead attention forward pass
		Args:
			q (torch.Tensor): The query tensor of shape (batch_size, seq_len, d_model).
			k (torch.Tensor): The key tensor of shape (batch_size, seq_len, d_model).
			v (torch.Tensor): The value tensor of shape (batch_size, seq_len, d_model).
			mask (torch.Tensor): The mask tensor of shape (batch_size, 1, seq_len, seq_len) or None.
		"""

		#step 1 - linear projection of input tensor q, k and v
		query = self.Wq(q)  # (batch, seq_len, d_model)
		key = self.Wk(k)    # (batch, seq_len, d_model)
		value = self.Wv(v)  # (batch, seq_len, d_model)

		# step 2 - reshape the query, key and value tensors to (batch, num_heads, seq_len, d_k)
		query = query.view(query.shape[0], query.shape[1], self.num_heads, self.d_k).transpose(1, 2)
		key = key.view(key.shape[0], key.shape[1], self.num_heads, self.d_k).transpose(1, 2)
		value = value.view(value.shape[0], value.shape[1], self.num_heads, self.d_k).transpose(1, 2)

		# step 3 - apply attention to the query, key and value tensors
		output, attention_weights = MultiHeadAttention.attention(query, key, value, mask, self.dropout)

		#step 4 - combine all heads together
		#(batch, num_heads, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
		output = output.transpose(1, 2).contiguous().view(output.shape[0], -1, self.d_model)

		#step 5 - apply the final linear projection to the output
		#(batch, seq_len, d_model) --> (batch, seq_len, d_model)
		return self.W0(output)  # (batch, seq_len, d_model)


class EncoderBlock(nn.Module):
	""" encoder block - this is the main component of the transformer model
		it consists of a multi-head self attention layer, a feed forward network and layer normalization.
		the input to the encoder block is the input embedding and positional encoding.
		the output of the encoder block is passed to the next encoder block or to the decoder block.
	"""
	def __init__(self, d_model: int = 512, num_heads: int = 8, d_ff: int = 2048, dropout: float = 0.1):
		"""
		Initializes the EncoderBlock module.
		Args:
			d_model (int): The dimension of the model (embedding size).
			num_heads (int): The number of attention heads.
			d_ff (int): The dimension of the feed forward network.
			dropout (float): Dropout rate.
		"""
		super(EncoderBlock, self).__init__()
		self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
		self.feed_forward = FeedForwardNetwork(d_model, d_ff, dropout)
		self.residual_attention = ResidualConnection(d_model, dropout)
		self.residual_feed_forward = ResidualConnection(d_model, dropout)

	def forward(self, x, mask=None):
		""" Forward pass for the encoder block.
		Args:
			x (torch.Tensor): The input tensor of shape (batch_size, seq_len, d_model).
			mask (torch.Tensor): The mask tensor of shape (batch_size, 1, seq_len, seq_len) or None.
		Returns:
			torch.Tensor: The output tensor after applying the encoder block.
		"""
		x = self.residual_attention(x, lambda x: self.self_attention(x, x, x, mask))
		x = self.residual_feed_forward(x, self.feed_forward)
		return x

class Encoder(nn.Module):
	""" given the number of layers Nx, the encoder is a stack of Nx encoder blocks.
		the input to the encoder is the input x and mask.
		the output of the encoder is passed to the decoder block.
	"""
	def __init__(self, features: int = 512, layers: nn.ModuleList = None):
		"""
		Initializes the Encoder module.
		Args:
			features (int): The dimension of the model (embedding size).
			layers (nn.ModuleList): A list of encoder blocks.
		"""
		super(Encoder, self).__init__()
		self.layers = layers
		self.norm = LayerNormalization(features)

	def forward(self, x, mask=None):
		""" Forward pass for the encoder.
		Args:
			x (torch.Tensor): The input tensor of shape (batch_size, seq_len, d_model).
			mask (torch.Tensor): The mask tensor of shape (batch_size, 1, seq_len, seq_len) or None.
		Returns:
			torch.Tensor: The output tensor after applying the encoder.
		"""
		for layer in self.layers:
			x = layer(x, mask)
		return self.norm(x)

class ResidualConnection(nn.Module):
	""" we have skip connections in all the Nx layers of both the encoder block and the decoder block.
		in the encoder block, there are residual connections around each of the sub-layers (multi-head attention and feed forward network),
		followed by layer normalization - output = LayerNorm(x + Sublayer(x)).
		in the decoder block, there are residual connections around each of the sub-layers (masked multi-head attention, multi-head attention and feed forward network),
	"""
	def __init__(self, features: int = 512, dropout: float = 0.1):
		"""
		Initializes the ResidualLayer module.
		Args:
			features (int): The dimension of the model (embedding size).
			dropout (float): Dropout rate.
		"""
		super(ResidualConnection, self).__init__()
		self.dropout = nn.Dropout(dropout)
		self.norm = LayerNormalization(features)

	def forward(self, x, sublayer):
		""" Applies the residual connection to the input tensor x and the sublayer.
		Args:
			x (torch.Tensor): The input tensor of shape (batch_size, seq_len, d_model).
			sublayer (nn.Module): The sublayer to which the residual connection will be applied.
		"""
		return x + self.dropout(sublayer(self.norm(x)))


class DecoderBlock(nn.Module):
	""" in the paper, the decoder block consists of 1) masked multihead attention + layer norm, 2) multihead attention + layer norm,
		and 3) feed forward network + layer norm.
		the masked multihead attention is used to prevent the decoder from attending to future tokens in the sequence.
		the multihead attention is used to attend to the encoder output. the queries come from the previous decoder block and
		the keys and values come from the encoder output. this allows every position in the decoder to attend to
		all positions in the input sentence.
	"""
	def __init__(self, d_model: int, num_heads: int = 8, d_ff: int = 2048, dropout: float = 0.1):
		"""
		Initializes the DecoderBlock module.
		Args:
			d_model (int): The dimension of the model (embedding size).
			num_heads (int): The number of attention heads.
			d_ff (int): The dimension of the feed forward network.
			dropout (float): Dropout rate.
		"""
		super(DecoderBlock, self).__init__()
		self.masked_attention = MultiHeadAttention(d_model, num_heads, dropout)	#masked multi-head attention
		self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)	#cross multi-head attention
		self.feed_forward = FeedForwardNetwork(d_model, d_ff, dropout)
		self.residual_masked_attention = ResidualConnection(d_model, dropout)
		self.residual_cross_attention = ResidualConnection(d_model, dropout)
		self.residual_feed_forward = ResidualConnection(d_model, dropout)

	def forward(self, x, encoder_output, encoder_mask, decoder_mask):
		""" Forward pass for the decoder block.
		Args:
			x (torch.Tensor): The input tensor of shape (batch_size, seq_len, d_model).
			encoder_output (torch.Tensor): The output tensor from the encoder of shape (batch_size, seq_len, d_model).
			encoder_mask (torch.Tensor): The mask tensor for the encoder output of shape (batch_size, 1, seq_len, seq_len) or None.
			decoder_mask (torch.Tensor): The mask tensor for the decoder input of shape (batch_size, 1, seq_len, seq_len) or None.
		"""
		x = self.residual_masked_attention(x, lambda x: self.masked_attention(x, x, x, decoder_mask))
		x = self.residual_cross_attention(x, lambda x: self.cross_attention(x, encoder_output, encoder_output, encoder_mask))
		x = self.residual_feed_forward(x, self.feed_forward)
		return x


class Decoder(nn.Module):
	""" using Nx number of layers, the decoder is a stack of Nx decoder blocks.
		the input to the decoder is the output from the encoder and the input to the decoder.
		the output of the decoder is passed to the final linear layer and softmax layer to get the output probabilities.
	"""
	def __init__(self, features: int = 512, layers: nn.ModuleList = None):
		"""
		Initializes the Decoder module.
		Args:
			features (int): The dimension of the model (embedding size).
			layers (nn.ModuleList): A list of decoder blocks.
		"""
		super(Decoder, self).__init__()
		self.layers = layers
		self.norm = LayerNormalization(features)

	def forward(self, x, encoder_output, encoder_mask=None, decoder_mask=None):
		""" Forward pass for the decoder.
		Args:
			x (torch.Tensor): The input tensor of shape (batch_size, seq_len, d_model).
			encoder_output (torch.Tensor): The output tensor from the encoder of shape (batch_size, seq_len, d_model).
			encoder_mask (torch.Tensor): The mask tensor for the encoder output of shape (batch_size, 1, seq_len, seq_len) or None.
			decoder_mask (torch.Tensor): The mask tensor for the decoder input of shape (batch_size, 1, seq_len, seq_len) or None.
		Returns:
			torch.Tensor: The output tensor after applying the decoder.
		"""
		for layer in self.layers:
			x = layer(x, encoder_output, encoder_mask, decoder_mask)
		return self.norm(x)


class ProjectionLayer(nn.Module):
	""" following the decoder output, we have a linear and softmax layer that maps the decoder output to the vocabulary size.
		this is the final layer of the transformer model.
	"""
	def __init__(self, d_model: int = 512, vocab_size: int = 10000):
		"""
		Initializes the ProjectionLayer module.
		Args:
			d_model (int): The dimension of the model (embedding size).
			vocab_size (int): The size of the vocabulary.
		"""
		super(ProjectionLayer, self).__init__()
		self.linear = nn.Linear(d_model, vocab_size)

	def forward(self, x):
		""" Forward pass for the projection layer. we apply a log softmax to the output of the linear layer.
			the output is the log probabilities of the vocabulary.
		Args:
			x (torch.Tensor): The input tensor of shape (batch_size, seq_len, d_model).
		Returns:
			torch.Tensor: The output tensor of shape (batch_size, seq_len, vocab_size) after applying the linear layer and log softmax.
		"""
		x = self.linear(x)
		return torch.log_softmax(x, dim=-1)  # (batch_size, seq_len, vocab_size)

class Transformer(nn.Module):
	""" combine the other classes to create the transformer model.
		the transformer model consists of an encoder, a decoder and a projection layer.
	"""
	def __init__(self, encoder: Encoder, decoder: Decoder, source_embedding: InputEmbedding, target_embedding: InputEmbedding,
				 source_positional_encoding: PositionalEncoding, target_positional_encoding: PositionalEncoding,
				 projection_layer: ProjectionLayer):
		"""
		Initializes the Transformer module.
		Args:
			encoder (Encoder): The encoder module.
			decoder (Decoder): The decoder module.
			source_embedding (InputEmbedding): The input embedding for the source sequence.
			target_embedding (InputEmbedding): The input embedding for the target sequence.
			source_positional_encoding (PositionalEncoding): The positional encoding for the source sequence.
			target_positional_encoding (PositionalEncoding): The positional encoding for the target sequence.
			projection_layer (ProjectionLayer): The projection layer to map the decoder output to the vocabulary size.
		"""
		super(Transformer, self).__init__()
		self.encoder = encoder
		self.decoder = decoder
		self.input_embedding = source_embedding
		self.output_embedding = target_embedding
		self.source_positional_encoding = source_positional_encoding
		self.target_positional_encoding = target_positional_encoding
		self.projection_layer = projection_layer

	def encode(self, src, src_mask=None):
		"""
		Encodes the input source sequence.
		Args:
			src (torch.Tensor): The input source tensor of shape (batch_size, seq_len).
			src_mask (torch.Tensor): The mask tensor for the source sequence of shape (batch_size, 1, seq_len, seq_len) or None.
		Returns:
			torch.Tensor: The encoded output tensor of shape (batch_size, seq_len, d_model).
		"""
		src = self.input_embedding(src)
		src = self.source_positional_encoding(src)
		return self.encoder(src, src_mask)

	def decode(self, tgt, encoder_output, src_mask=None, tgt_mask=None):
		"""
		Decodes the target sequence using the encoder output.
		Args:
			tgt (torch.Tensor): The input target tensor of shape (batch_size, seq_len).
			encoder_output (torch.Tensor): The output tensor from the encoder of shape (batch_size, seq_len, d_model).
			src_mask (torch.Tensor): The mask tensor for the source sequence of shape (batch_size, 1, seq_len, seq_len) or None.
			tgt_mask (torch.Tensor): The mask tensor for the target sequence of shape (batch_size, 1, seq_len, seq_len) or None.
		Returns:
			torch.Tensor: The decoded output tensor of shape (batch_size, seq_len, d_model).
		"""
		tgt = self.output_embedding(tgt)
		tgt = self.target_positional_encoding(tgt)
		return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

	def project(self, decoder_output):
		"""
		Projects the decoder output to the vocabulary size.
		Args:
			decoder_output (torch.Tensor): The output tensor from the decoder of shape (batch_size, seq_len, d_model).
		Returns:
			torch.Tensor: The projected output tensor of shape (batch_size, seq_len, vocab_size).
		"""
		return self.projection_layer(decoder_output)


#build_transformer function to create the transformer model
#inputs to function - source vocab size, target vocab size, source sequence length, target sequence length, d_model, num_layers, num_heads, d_ff, dropout. 
#returns the Transformers class object
def build_transformer(src_vocab_size, tgt_vocab_size, src_seq_len, tgt_seq_len, d_model=512, num_layers=6, num_heads=8, d_ff=2048, dropout=0.1):
	"""
	Creates a Transformer model with encoder, decoder and projection layers. use the Transformers class to create the model 
	and initialise the parameters with xavier initialization.
	Args:
		src_vocab_size (int): The size of the source vocabulary.
		tgt_vocab_size (int): The size of the target vocabulary.
		src_seq_len (int): The maximum length of the source sequences.
		tgt_seq_len (int): The maximum length of the target sequences.
		d_model (int): The dimension of the model (embedding size).
		num_layers (int): The number of encoder and decoder layers.
		num_heads (int): The number of attention heads.
		d_ff (int): The dimension of the feed forward network.
		dropout (float): Dropout rate.
	Returns:
		Transformer: An instance of the Transformer model.
	"""
	# Create input and output embeddings
	source_embedding = InputEmbedding(d_model, src_vocab_size)
	target_embedding = InputEmbedding(d_model, tgt_vocab_size)

	# Create positional encodings
	source_positional_encoding = PositionalEncoding(d_model, src_seq_len, dropout)
	target_positional_encoding = PositionalEncoding(d_model, tgt_seq_len, dropout)

	# Create encoder layers
	encoder_layers = nn.ModuleList([EncoderBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

	# Create decoder layers
	decoder_layers = nn.ModuleList([DecoderBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

	# Create projection layer
	projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

	# Create the Transformer model - FIXED: Use keyword arguments for clarity
	transformer = Transformer(
		Encoder(features=d_model, layers=encoder_layers),
		Decoder(features=d_model, layers=decoder_layers),
		source_embedding,
		target_embedding,
		source_positional_encoding,
		target_positional_encoding,
		projection_layer
	)

	#initialise the parameters of the model using xavier initialization
	for p in transformer.parameters():
		if p.dim() > 1:
			nn.init.xavier_uniform_(p)

	return transformer