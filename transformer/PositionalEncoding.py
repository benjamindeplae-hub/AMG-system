import torch
import torch.nn as nn
import math

# https://www.youtube.com/watch?v=zxQyTK8quyY&t=1641s
# https://www.datacamp.com/tutorial/building-a-transformer-with-py-torch?

# 1. Parameter management
# Any layers you define inside (like nn.Linear) are automatically:
# - Registered as trainable parameters
# - Included when you call model.parameters()
# This is what allows optimizers like Adam to update weights
# 2. Built-in methods you get for free
# By inheriting from nn.Module, your class gains useful methods like:
# - model.forward(x) -> defines computation
# - model(x) -> automatically calls forward
# - model.train() -> sets training mode
# - model.eval() -> sets evaluation mode
# - model.to(device) -> moves model to GPU/CPU
# - model.state_dict() -> saves weights
# 3. Forward pass definition
# You define how data flows through your module by implementing
# def forward(self, x):
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        # d_model = dimension of word embedding
        # max_seq_length = max supported length of sequence
        super().__init__()

        # Attention Is All You Need by Ashish Vaswani
        # The Original Positional Encoding Formula
        # In the paper the encoding is defined as:
        # PE(pos, 2i) = sin( pos / 10000^(2i/d_model) )
        # PE(pos, 2i + 1) = cos( pos / 10000^(2i/d_model) )
        # Where:
        # pos = position in the sequence
        # i = embedding dimension index
        # d_model = embedding size (e.g., 512)
        # Even dimensions use sin, odd dimensions use cos

        # 10000 ** (2*i/d_model)
        # Directly computing powers can cause numerical instability
        # Reasons:
        # - exponentiation can overflow
        # - repeated powers are expensive
        # - GPUs prefer vectorized operations
        # So the formula is rewritten using logarithms and exponentials

        # 1. Start 10000^(2i/d_model)
        # a^b = e^(log(a^b))
        # log(a^b) = b * log(a)
        # So:
        # a^b = e^(b*log(a))
        # So:
        # 10000^(2i/d_model) = e^((2i/d_model) * log(10000))
        # Now invert it because the formula divides by it:
        # 1/a = a^(-1)
        # So:
        # 1/10000^(2i/d_model) = e^-((2i/d_model) * log(10000))
        # So the argument becomes:
        # pos * e^(−(2i/d_model) * log(10000))

        # pe = zero matrix [max_seq_length, d_model]
        pe = torch.zeros(max_seq_length, d_model)

        # positions = [0, 1, ..., max_seq_length] 
        # torch.unsqueeze(x, 1) = [[0], [1], ..., [max_seq_length]]
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)

        # torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        # = e^([0, 2, 4, ..., d_model] * -(log(10000) / d_model))
        # = e^(2i * -(log(10000) / d_model))
        # = e^(−(2i/d_model) * log(10000))

        # If d_model = 8:
        # log(10000) ≈ 9.21034
        # So:
        # -(log(10000)/8) ≈ -1.15129
        # Now multiply:
        # [0, 2, 4, 6] * -1.15129
        # So:
        # [0, -2.3026, -4.6052, -6.9078]
        # Now we exponentiate each element:
        # exp(0) = 1
        # exp(-2.3026) ≈ 0.1
        # exp(-4.6052) ≈ 0.01
        # exp(-6.9078) ≈ 0.001
        # So:
        # [1.0, 0.1, 0.01, 0.001]
        # The tensor is 1-dimensional of length (d_model / 2)
        # So the vector looks like a geometric decay:
        # [1, 1/10000^(2/d_model), 1/10000^(4/d_model), 1/10000^(6/d_model), ...]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        # position * div_term 
        # = pos * e^(−(2i/d_model) * log(10000))
        # = pos / 10000^(2i/d_model)
        # Python slicing format: start : stop : step
        # rows: : = all rows
        # columns: 0::2 = start at index 0, go to the end, step by 2
        # Even dimensions -> sin
        # Odd dimensions -> cos
        
        # Example:
        # d_model = 4
        # max_seq_length = 5
        # position = [[0], [1], [2], [3], [4]]
        # div_term ≈ [1.0, 0.01]
        # position * div_term = 
        #   [[0*1  0*0.01]
        #    [1*1  1*0.01]
        #    [2*1  2*0.01]
        #    [3*1  3*0.01]
        #    [4*1  4*0.01]]
        # = [[0  0]
        #    [1  0.01]
        #    [2  0.02]
        #    [3  0.03]
        #    [4  0.04]]
        # pe =
        # | position | dim0 (sin) | dim1 (cos) | dim2 (sin)    | dim3 (cos)    |
        # | -------- | ---------- | ---------- | ------------- | ------------- |
        # | 0        | sin(0 * 1) | cos(0 * 1) | sin(0 * 0.01) | cos(0 * 0.01) |
        # | 1        | sin(1 * 1) | cos(1 * 1) | sin(1 * 0.01) | cos(1 * 0.01) |
        # | 2        | sin(2 * 1) | cos(2 * 1) | sin(2 * 0.01) | cos(2 * 0.01) |
        # | 3        | sin(3 * 1) | cos(3 * 1) | sin(3 * 0.01) | cos(3 * 0.01) |
        # | 4        | sin(4 * 1) | cos(4 * 1) | sin(4 * 0.01) | cos(4 * 0.01) |
        # =
        # [[ 0.0000,  1.0000, 0.0000, 1.0000],
        #  [ 0.8415,  0.5403, 0.0100, 0.9999],
        #  [ 0.9093, -0.4161, 0.0200, 0.9998],
        #  [ 0.1411, -0.9900, 0.0300, 0.9996],
        #  [-0.7568, -0.6536, 0.0400, 0.9992]]
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # pe.unsqueeze(0)
        # adds a batch dimension:
        # shape = (1, 5, 4)
        # [
        #   [
        #     [0.0000, 1.0000, 0.0000, 1.0000],
        #     [0.8415, 0.5403, 0.0100, 0.9999],
        #     [0.9093,-0.4161, 0.0200, 0.9998],
        #     [0.1411,-0.9900, 0.0300, 0.9996],
        #     [-0.7568,-0.6536,0.0400,0.9992]
        #   ]
        # ]
        # The 1 is a “batch dimension” added with unsqueeze(0) so it can broadcast across any real batch size

        # When you build a nn.Module (like your PositionalEncoding), there are two main types of things you can store inside it:
        # 1. Parameters (nn.Parameter)
        # - These are trainable
        # - They appear in model.parameters()
        # - They get updated by the optimizer during training
        # 2. Buffers (register_buffer)
        # - These are non-trainable tensors that are part of the module
        # - They move with the model when you call .to(device) or .cuda()
        # - They are saved in state_dict() so they get saved/loaded with your model
        # - They do not get gradients or update during training
        # In short:
        # Buffers are constants or “state” tensors that the model uses, but the optimizer doesn’t change them
        # Why positional encoding is a buffer?
        # pe is deterministic positional encoding, it doesn’t change during training
        # Move to GPU if the model moves (model.to('cuda'))
        # Be saved/loaded automatically with the model
        # If we made it a Parameter, the optimizer would try to update it, which we don’t want
        # Adds a tensor attribute to your module: self.pe now exists
        # Tells PyTorch: “this is part of the module state, but not a trainable parameter.”
        # Makes it show up in state_dict(), but not in parameters()
        self.register_buffer('pe', pe.unsqueeze(0))
    
    # Gives the option of self.model(x) for an easy forward pass through the model
    def forward(self, x):
        # input x has shape: (batch_size, seq_len, d_model)
        # Example: x.shape = (32, 10, 512)
        # self.pe has shape: (1, max_seq_length, d_model)
        # Example: x.shape = (1, 5000, 512)
        # x.size(1) = 10 = sequence length of the input
        # So the slice becomes: self.pe[:, :10]
        # Return shape of self.pe: (1, 10, d_model)
        # This ensures the positional encoding matches the length of the input
        return x + self.pe[:, :x.size(1)]
