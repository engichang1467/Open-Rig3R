import torch 
import torch.nn as nn
import torch.nn.functional as F

class BaseHead(nn.Module):
    """
        Base head for token-to-map prediction.
    """
    def __init__(self, in_dim, out_dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or in_dim

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, tokens):
        """
            Args:
                tokens: (B, N, C) decoder output tokens
            Returns:
                pred: (B, N, out_dim)
        """
        return self.mlp(tokens)
    