import torch
import torch.nn as nn
import math

# https://www.youtube.com/watch?v=zxQyTK8quyY&t=1641s
# https://www.datacamp.com/tutorial/building-a-transformer-with-py-torch?

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        # d_model = dimension of word embedding
        # num_heads is what turns basic attention into multi-head attention
        # num_heads = number of parallel attention mechanisms (heads) running at the same time
        # Instead of doing attention once, the model does it multiple times in parallel, each with its own perspective
        # A single attention operation can only focus on one type of relationship at a time
        # With multiple heads, the model can learn different things simultaneously, like:
        # Head 1 -> grammatical relationships
        # Head 2 -> long-distance dependencies
        # Head 3 -> semantic meaning    
        # Head 4 -> positional patterns
        super().__init__()
        # Ensure that the model dimension (d_model) is divisible by the number of heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        # Initialize dimensions
        self.d_model = d_model # Model's dimension
        self.num_heads = num_heads # Number of attention heads
        self.d_k = d_model // num_heads # Dimension of each head's key, query, and value
        
        # Linear layers for transforming inputs
        self.W_q = nn.Linear(d_model, d_model) # Query transformation
        self.W_k = nn.Linear(d_model, d_model) # Key transformation
        self.W_v = nn.Linear(d_model, d_model) # Value transformation
        self.W_o = nn.Linear(d_model, d_model) # Output transformation
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # Calculate attention scores
        """
        # Attention Is All You Need by Ashish Vaswani
        # The Original Attention Formula
        # Attention(Q, K, V) = softmax( QK^T / d_k^(1/2) ) * V

        # Example:
        # batch_size = 1
        # seq_len = 2
        # d_model = 4
        # num_heads = 2
        # d_k = d_model / num_heads = 2
        
        # Q, K, V shape's = (batch=1, seq_len=2, d_model=4)
        # Q = [
        #   [1, 0, 1, 1],
        #   [0, 1, 1, 0]
        # ]
        # K = [
        #   [1, 2, 0, 1],
        #   [3, 4, 1, 0]
        # ]
        # V = [
        #   [10, 0, 5, 5],
        #   [0, 10, 10, 0]
        # ]

        # Step 1 Seperate Heads:
        # Q, K, V shape's = (batch=1, heads=2, seq_len=2, d_k=2)
        # Q = [
        #   head 1: [[1, 0],
        #            [0, 1]],
        #   head 2: [[1, 1],
        #            [1, 0]]
        # ]
        # K = [
        #   head 1: [[1, 2],
        #            [3, 4]],
        #   head 2: [[0, 1],
        #            [1, 0]]
        # ]
        # V = [
        #   head 1: [[10, 0],
        #            [0, 10]],
        #   head 2: [[5, 5],
        #            [10, 0]]
        # ]

        # Step 2 Compute Attention Scores:
        # Q @ K^T = (1, 2, 2, 2) @ (1, 2, 2, 2) = (1, 2, 2, 2)
        # We compute per head independently:

        # Head 1
        # Q1 = [[1, 0], 
        #       [0, 1]]
        # K1^T = [[1, 3],
        #         [2, 4]]
        # Q1 @ K1^T = [[1, 3],
        #              [2, 4]]
        # Scaled by d_k^(1/2) ≈ 1.41
        # [[0.71, 2.12],
        #  [1.41, 2.83]]
        # Softmax (row-wise):
        # [[0.20, 0.80],
        #  [0.20, 0.80]]

        # Head 2
        # Q1 = [[1, 1],
        #       [1, 0]]
        # K2^T = [[0, 1],
        #         [1, 0]]
        # Q2 @ K2^T = [[1, 1],
        #              [0, 1]]
        # Scaled by d_k^(1/2) ≈ 1.41
        # [[0.71, 0.71],
        #  [0.00, 0.71]]
        # Softmax (row-wise):
        # [[0.50, 0.50],
        #  [0.33, 0.67]]

        # Final result:
        # [
        #   head 1: [[0.20, 0.80],
        #            [0.20, 0.80]],
        #   head 2: [[0.50, 0.50],
        #            [0.33, 0.67]]
        # ]
        # shape = (batch=1, heads=2, seq_len=2, d_k=2)

        # Step 3 Apply Attention to V:
        # attn_probs @ V = (1, 2, 2, 2) @ (1, 2, 2, 2) = (1, 2, 2, 2)
        
        # Head 1
        # Weights = [[0.20, 0.80],
        #            [0.20, 0.80]]
        # V1 = [[10, 0],
        #       [0, 10]]
        # Row 1: 0.20*[10,0] + 0.80*[0,10] = [2, 8]
        # Row 2: same -> [2, 8]

        # Head 2
        # Weights = [[0.50, 0.50],
        #            [0.33, 0.67]]
        # V1 = [[5, 5],
        #       [10, 0]]
        # Row 1: 
        # 0.5*[5,5] + 0.5*[10,0] 
        # = [2.5,2.5] + [5,0] 
        # = [7.5, 2.5]
        # Row 2: 
        # 0.33*[5,5] + 0.67*[10,0] 
        # ≈ [1.65,1.65] + [6.7,0]
        # ≈ [8.35, 1.65]

        # Final result:
        # [
        #   head 1: [[2, 8],
        #            [2, 8]],
        #   head 2: [[7.5, 2.5],
        #            [8.35, 1.65]]
        # ]
        # shape = (batch=1, heads=2, seq_len=2, d_k=2)

        # Step 4 Combine Heads:
        # [
        #   [2, 8, 7.5, 2.5],
        #   [2, 8, 8.35, 1.65]
        # ]
        # shape = (batch=1, seq_len=2, d_model=4)
        """

        # K.transpose(-2, -1) swaps the last two axes of K
        # (batch_size, num_heads, seq_length, d_k) -> (batch_size, num_heads, d_k, seq_length)

        # torch.matmul(A, B) performs matrix multiplication:
        # It takes rows of A and columns of B, and computes dot products
        # Example:
        # A = [[1, 2],
        #      [3, 4]]
        # B = [[5, 6],
        #      [7, 8]]
        # Then:
        # A @ B =
        # [[1*5 + 2*7,   1*6 + 2*8],
        #  [3*5 + 4*7,   3*6 + 4*8]]
        # For torch.matmul(A, B) with shapes:
        # A: (..., m, n)
        # B: (..., n, p)
        # The result shape is (..., m, p)
        # torch.matmul does NOT do one giant high-dimensional multiplication
        # It treats all leading dimensions as batch dimensions and performs many independent 2D matrix multiplications
        # So torch.matmul(Q, K^T) becomes:
        # For every batch and every head, do a normal matrix multiplication
        # torch.matmul(Q, K.transpose(-2, -1)) = (batch_size, num_heads, seq_length, d_k) @ (batch_size, num_heads, d_k, seq_length)
        # attn_scores.shape = (batch_size, num_heads, seq_length, seq_length)
        """
        # Parallelism insight:

        # torch.matmul(Q, K^T) does NOT do one big single matrix multiplication
        # Instead, it behaves like MANY smaller matrix multiplications happening at the same time

        # Shapes:
        # Q = (batch_size, num_heads, seq_length, d_k)
        # K^T = (batch_size, num_heads, d_k, seq_length)

        # Conceptually, you can think of it like this:

        # for batch in range(batch_size):
        #     for head in range(num_heads):
        #         attn_scores[batch, head] = Q[batch, head] @ K[batch, head].T

        # So if:
        # batch_size = 32
        # num_heads = 8
        # Then we are doing 32 * 8 = 256 matrix multiplications

        # BUT:
        # PyTorch does this ALL AT ONCE using tensor operations (no Python loops)
        # On a GPU, these 256 multiplications are executed in parallel

        # This is where the "multi-head" in attention becomes truly parallel:
        # Each head is completely independent and computed simultaneously
        """
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply mask if provided (useful for preventing attention to certain parts like padding)
        # Sometimes we don’t want the model to attend to certain positions:
        # - Padding tokens: We don’t want attention to go to padded parts of the sequence
        # - Causal masking (decoder): Prevent attention to future tokens in language modeling
        # mask has same shape as attention (or broadcastable to it)
        # Usually: 1 = keep, 0 = mask out
        if mask is not None:
            # masked_fill(condition, value) replaces elements where condition is True with value
            # Wherever mask == 0, we set that attn_score to -1e9 (a very large negative number)
            # Why -1e9?
            # Because next we apply softmax:
            # If x_i = -1e9, then e^{x_i} ≈ 0
            # This effectively removes that position from contributing to attention
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        # Softmax is applied to obtain attention probabilities
        # Softmax converts raw scores into probabilities that sum to 1 along a certain axis
        attn_probs = torch.softmax(attn_scores, dim=-1)
        
        # Multiply by values to obtain the final output
        # (batch_size, num_heads, seq_length, seq_length) @ (batch_size, num_heads, seq_length, d_k)
        # output.shape = (batch_size, num_heads, seq_length, d_k)
        output = torch.matmul(attn_probs, V)
        return output
        
    def split_heads(self, x):
        # Reshape the input to have num_heads for multi-head attention
        batch_size, seq_length, d_model = x.size()
        # x.view(batch_size, seq_length, self.num_heads, self.d_k)
        # self.num_heads = number of attention heads (e.g., 8)
        # self.d_k = dimension per head (d_model // num_heads)
        # This reshapes the tensor to separate the embedding dimension into multiple heads:
        # x.view(batch_size, seq_length, num_heads, d_k)
        # Example:
        # x.shape = (32, 10, 512)
        # d_model = 512, num_heads = 8 -> d_k = 512 / 8 = 64
        # So new shape: (32, 10, 8, 64)

        # .transpose(1, 2)
        # Swaps dimension 1 (seq_length) with dimension 2 (num_heads)
        # Original after view: (batch_size, seq_length, num_heads, d_k) -> (32, 10, 8, 64)
        # After transpose(1, 2): (batch_size, num_heads, seq_length, d_k) -> (32, 8, 10, 64)
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        
    def combine_heads(self, x):
        # Combine the multiple heads back to original shape
        batch_size, _, seq_length, d_k = x.size()
        # .transpose(1, 2)
        # Swaps dimension 1 (num_heads) with dimension 2 (seq_length)
        # Original after view: (batch_size, num_heads, seq_length, d_k) -> (32, 10, 8, 64)
        # After transpose(1, 2): (batch_size, seq_length, num_heads, d_k) -> (32, 8, 10, 64)

        # In PyTorch, after a transpose, the tensor may not be contiguous in memory
        # .contiguous() ensures that we can safely view() it into a new shape without errors

        # .view(batch_size, seq_length, self.d_model)
        # self.d_model = num_heads * d_k
        # Concatenates all the heads back along the embedding dimension
        # Shape becomes (batch_size, seq_length, d_model) again
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        
    def forward(self, Q, K, V, mask=None):
        # Q = K = V = Input x = (batch_size, seq_len, d_model)
        # Apply linear transformations and split heads
        Q = self.split_heads(self.W_q(Q))  # Result: (batch_size, num_heads, seq_length, d_k)
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        
        # Perform scaled dot-product attention
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)  # Result: (batch_size, num_heads, seq_length, d_k)
        
        # Combine heads and apply output transformation
        output = self.W_o(self.combine_heads(attn_output))  # Result: (batch_size, seq_len, d_model)
        return output
