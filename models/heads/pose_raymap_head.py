import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_head import BaseHead


class PoseRaymapHead(BaseHead):
    """
        Predicts per-patch 3D ray directions in pose-relative (camera) coordinates.
        Typically used to regress a normalized direction vector for each patch.
    """
    def __init__(self, in_dim=1024, hidden_dim=512, normalize=True):
        super().__init__(in_dim, out_dim=3, hidden_dim=hidden_dim)
        self.normalize = normalize
    
    def forward(self, tokens):
        """
            Args:
                tokens: (B, N, C)
            Returns:
                rays: (B, N, 3) unit ray directions
        """
        rays = super().forward(tokens)

        # Normalize rays to unit vectors (optional)
        if self.normalize:
            rays = F.normalize(rays, dim=-1)

        return rays
    
    ## (Optional) to condition the raymap on known intrinsics or pixel grid
    # def forward(self, tokens, ray_grid=None):
    #     rays = super().forward(tokens)
    #     if ray_grid is not None:
    #         rays = rays + ray_grid  # residual correction
    #     return F.normalize(rays, dim=-1)