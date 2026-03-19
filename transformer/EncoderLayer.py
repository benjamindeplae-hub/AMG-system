import torch.nn as nn
from MultiHeadAttention import MultiHeadAttention
from PositionWiseFeedForward import PositionWiseFeedForward

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        # Instead of normalizing across a batch (like BatchNorm), LayerNorm normalizes each individual data point independently
        # - It looks at all the features of one sample
        # - Computes mean and variance
        # - Normalizes using those values
        # For an input vector x:
        # LayerNorm(x) = (x - μ) / (σ^2 + ε)^(1/2)  * γ + β
        # Where:
        # - μ = mean of the features
        # - σ^2 = variance of the features
        # - γ, β = learnable parameters (scale and shift)
        # - ϵ = small value for numerical stability
        # Why not reuse the same LayerNorm?
        # Because each LayerNorm has its own learnable parameters:
        # γ (scale), β (shift)
        # These parameters adapt to different distributions:
        # Attention output ≠ Feed-forward output
        # So sharing one LayerNorm would:
        # - Reduce flexibility
        # - Hurt performance
        # LayerNorm(x) = normalize(x) * γ + β
        # That means it’s not just: "make mean = 0, variance = 1"
        # It’s: "normalize, then reshape the distribution in a learnable way"
        # Reusing forces the model to say:
        # "I will scale/shift attention outputs and feed-forward outputs in the same way"
        # Which is restrictive because:
        # - Attention might need amplification in some dimensions
        # - Feed-forward might need suppression in those same dimensions
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        # Residual Connections
        x = self.norm1(x + self.dropout(attn_output))
        # Feed foward = "What have I learned now?"
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x
