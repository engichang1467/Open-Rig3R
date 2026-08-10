import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_head import BaseHead


class RaymapHead(BaseHead):
    """
        Rig3R sec 3.3: "Each raymap head consists of two MLPs: one predicts per-pixel
        ray directions, and the other predicts a global camera center via average
        pooling over patch tokens."

        Sec 3.1 defines a raymap as a unit direction per pixel with all rays sharing
        one camera center, so the center is pooled per frame rather than predicted
        per patch. Pooling instead of a query token is what keeps every gradient
        flowing through the patch tokens.

        The pose-relative and rig-relative heads are the same module with their own
        weights; sec 3.3 shares weights across frames, not across heads.
    """
    def __init__(self, in_dim=1024, hidden_dim=512):
        super().__init__(in_dim, out_dim=3, hidden_dim=hidden_dim)
        self.center_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, tokens):
        """
            Args:
                tokens: (B, V, P, C) decoder tokens, grouped by frame
            Returns:
                raymap: (B, V, P, 6) the frame's center broadcast over its patches,
                        concatenated with per-patch unit ray directions
                center: (B, V, 3) one camera center per frame
        """
        directions = F.normalize(super().forward(tokens), dim=-1)  # (B, V, P, 3)
        center = self.center_mlp(tokens.mean(dim=2))               # (B, V, 3)

        raymap = torch.cat([center.unsqueeze(2).expand_as(directions), directions], dim=-1)
        return raymap, center
