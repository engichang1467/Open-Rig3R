import torch
import torch.nn.functional as F
from .base_head import BaseHead


class RigRaymapHead(BaseHead):
    """
        Predicts per-patch rig-relative rays (origin, direction) from patch tokens.
        Rig pose reaches the model only through the metadata embedding, never here.
    """
    def __init__(self, in_dim=1024, hidden_dim=512, normalize=False):
        super().__init__(in_dim, out_dim=6, hidden_dim=hidden_dim)
        self.normalize = normalize

    def forward(self, tokens):
        """
            Args:
                tokens: (B, N, C)
            Returns:
                rig_rays: (B, N, 6) rig-relative rays (origin, direction)
        """
        rays = super().forward(tokens) # (B, N, 6)

        # Normalize direction vectors, leaving origins untouched
        if self.normalize:
            rays = torch.cat([rays[..., :3], F.normalize(rays[..., 3:], dim=-1)], dim=-1)

        return rays
