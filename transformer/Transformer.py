import torch
import torch.nn as nn
from PositionalEncoding import PositionalEncoding
from EncoderLayer import EncoderLayer
from DecoderLayer import DecoderLayer

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        super().__init__()
        # Word embedding
        # src_vocab_size = the number of unique items, vocabulary size
        # d_model = the size of each embedding vector
        # nn.Embedding is a lookup table that maps discrete indices (usually integers representing words or tokens) to dense vectors (embeddings)
        # Suppose src_vocab_size = 10,000 and d_model = 512
        # nn.Embedding(10000, 512) will create a matrix of size (10000, 512)
        # Each row corresponds to a token index (from 0 to 9999), and the row contains a 512-dimensional embedding vector
        # When you input a tensor of indices like:
        # input = torch.tensor([2, 5, 8])
        # The embedding layer will return:
        # output = embedding(input) # output.shape -> (3, 512)
        # The embeddings are learnable parameters. During training, the model updates them so that tokens with similar contexts end up with similar vectors
        # nn.Embedding(src_vocab_size, d_model) turns integer token indices into d_model-dimensional vectors, which the model can then use as inputs for further layers like a Transformer
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)

        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        # Each layer builds on the previous one. Early layers capture low-level patterns (like local token relationships), 
        # and deeper layers capture high-level patterns (like semantic meaning or long-range dependencies)
        # If you only had one layer, the model would be shallow and could only capture very simple relationships
        # For example:
        # A single encoder layer can understand "this word relates to that word" in isolation
        # Multiple encoder layers can understand "this phrase relates to that other phrase in the sentence" and "the overall context of the paragraph."
        # Think of it like a stack of experts:
        # Layer 1: Understands simple words
        # Layer 2: Understands phrases
        # Layer 3: Understands sentences
        # Layer 4+: Understands context across multiple sentences
        # The same idea applies to the decoder: each layer progressively refines the output prediction using more context
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src, tgt):
        # (src != 0)
        # Checks each token in the source sequence
        # Usually, 0 represents padding
        # Result: a boolean tensor where True means it’s a real token, False means padding

        # .unsqueeze(1).unsqueeze(2)
        # Adds two singleton dimensions for broadcasting in multi-head attention
        # If src shape is (batch_size, src_len), after this it becomes (batch_size, 1, 1, src_len)
        # This mask will be used so the model ignores padding tokens in the source
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        # (tgt != 0) = boolean mask for real tokens vs padding
        # .unsqueeze(1).unsqueeze(3)
        # If tgt shape is (batch_size, tgt_len), now it becomes (batch_size, 1, tgt_len, 1) for broadcasting in attention
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        
        seq_length = tgt.size(1)
        # torch.ones(1, seq_length, seq_length) creates a square matrix of ones: (1, tgt_len, tgt_len)
        # torch.triu(..., diagonal=1) creates upper triangular matrix (everything above main diagonal = 1, diagonal & below = 0)
        # 1 - ... -> flips it: upper triangle becomes 0, diagonal and lower triangle = 1
        # .bool() -> converts to boolean mask: True means allowed, False means masked
        # This mask ensures the decoder cannot attend to future tokens, which is why it’s called a no-peek or causal mask
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
        # tgt_mask from padding is combined (&) with nopeak_mask
        # Result: True only if a token is real and not in the future
        # Shape: (batch_size, 1, tgt_len, tgt_len) -> ready for attention layers
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask

    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        output = self.fc(dec_output)
        return output