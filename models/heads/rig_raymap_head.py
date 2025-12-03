import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_head import BaseHead


class RigRaymapHead(BaseHead):
    """
        Predicts per-patch 3D ray directions in rig-relative coordinates.
        Optionally applies camera-to-rig transformation if provided.
    """
    def __init__(self, in_dim=1024, hidden_dim=512, normalize=False):
        super().__init__(in_dim, out_dim=6, hidden_dim=hidden_dim)
        self.normalize = normalize

    def forward(self, tokens, cam2rig=None):
        """
            Args:
                tokens: (B, N, C)
                cam2rig: optional (B, 3, 3) or (B, 4, 4) rotation/transform matrices
            Returns:
                rig_rays: (B, N, 6) rig-relative rays (origin, direction)
        """
        rays = super().forward(tokens) # (B, N, 6)

        # Split into origin and direction
        origins = rays[..., :3]
        directions = rays[..., 3:]

        # Transform from camera → rig frame if extrinsics given
        if cam2rig is not None:
            if cam2rig.dim() == 4:  # (B, V, 3, 3) or (B, V, 4, 4)
                B, V, _, _ = cam2rig.shape
                N = rays.shape[1]
                assert N % V == 0, f"Expected N divisible by frames (V), got N={N}, V={V}"
                patches_per_frame = N // V

                # slice rotation and translation
                R = cam2rig[..., :3, :3]  # (B, V, 3, 3)
                t = cam2rig[..., :3, 3] if cam2rig.shape[-1] == 4 else None # (B, V, 3)

                origins = origins.view(B, V, patches_per_frame, 3)
                directions = directions.view(B, V, patches_per_frame, 3)
                
                # Apply rotation
                origins = torch.einsum('bvij,bvnj->bvni', R, origins)
                directions = torch.einsum('bvij,bvnj->bvni', R, directions)

                # Apply translation to origin if available
                if t is not None:
                    origins = origins + t.unsqueeze(2)

                origins = origins.reshape(B, N, 3)
                directions = directions.reshape(B, N, 3)

            elif cam2rig.dim() == 3:  # (B, 3, 3) or (B, 4, 4)
                R = cam2rig[..., :3, :3]
                t = cam2rig[..., :3, 3] if cam2rig.shape[-1] == 4 else None
                
                origins = torch.einsum('bij,bnj->bni', R, origins)
                directions = torch.einsum('bij,bnj->bni', R, directions)

                if t is not None:
                    origins = origins + t.unsqueeze(1)
            else:
                raise ValueError(f"Unexpected cam2rig shape: {cam2rig.shape}")

        # Normalize direction vectors
        if self.normalize:
            directions = F.normalize(directions, dim=-1)

        # Concatenate back
        rays = torch.cat([origins, directions], dim=-1)

        # sanity check
        assert rays.shape[-1] == 6, f"Unexpected ray output shape: {rays.shape}"

        return rays

