import torch 
import torch.nn as nn

from .base_head import BaseHead

class PointMapHead(BaseHead):
    """
        Predicts 3D pointmaps (XYZ per patch)
    """
    def __init__(self, in_dim=1024, hidden_dim=512):
        super().__init__(in_dim, out_dim=3, hidden_dim=hidden_dim)
    
    def forward(self, tokens):
        """
            Args:
                tokens: (B, N, C)
            Returns:
                points: (B, N, 3)
        """
        points = super().forward(tokens)
        return points