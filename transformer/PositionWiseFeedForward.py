import torch.nn as nn

# https://www.datacamp.com/tutorial/building-a-transformer-with-py-torch?

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        # d_model = dimension of word embedding
        # d_ff = dimension of the hidden layer inside the feed-forward network
        # Each token embedding (size d_model) is:
        # Expanded to a larger space (d_ff)
        # Passed through a non-linearity (ReLU)
        # Compressed back to d_model
        # To give the model more capacity to learn complex patterns
        # If you stayed in d_model, the transformation would be too limited
        # By going to a higher dimension:
        # The model can learn richer representations
        # In the original transformer paper:
        # d_model = 512, d_ff = 2048
        # So d_ff is usually much larger than d_model (often 4× bigger)
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        # Relu activation function to catch non-linearity
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))
