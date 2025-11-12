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
        super().__init__(in_dim, out_dim=3, hidden_dim=hidden_dim)
        self.normalize = normalize

    def forward(self, tokens, cam2rig=None):
        """
            Args:
                tokens: (B, N, C)
                cam2rig: optional (B, 3, 3) or (B, 4, 4) rotation/transform matrices
            Returns:
                rig_rays: (B, N, 3) normalized rig-relative ray directions
        """
        rays = super().forward(tokens) # (B, N, 3)

        # Transform from camera → rig frame if extrinsics given
        if cam2rig is not None:
            # if cam2rig.shape[-1] == 4:  # full 4x4
            #     R = cam2rig[:, :3, :3]
            # else:
            #     R = cam2rig
            # rays = torch.einsum('bij,bnj->bni', R, rays)


            if cam2rig.dim() == 4:  # (B, V, 3, 3) or (B, V, 4, 4)
                B, V, _, _ = cam2rig.shape
                N = rays.shape[1]
                assert N % V == 0, f"Expected N divisible by frames (V), got N={N}, V={V}"
                patches_per_frame = N // V

                # slice rotation
                R = cam2rig[..., :3, :3]  # always (B, V, 3, 3)
                rays = rays.view(B, V, patches_per_frame, 3)
                rays = torch.einsum('bvij,bvnj->bvni', R, rays)  # (B, V, P, 3)
                rays = rays.reshape(B, N, 3)
            elif cam2rig.dim() == 3:  # (B, 3, 3) or (B, 4, 4)
                R = cam2rig[..., :3, :3]
                rays = torch.einsum('bij,bnj->bni', R, rays)
            else:
                raise ValueError(f"Unexpected cam2rig shape: {cam2rig.shape}")

        # Normalize rays
        if self.normalize:
            # print("[DEBUG] pre-norm stats:", rays.mean().item(), rays.std().item())
            rays = F.normalize(rays, dim=-1)

        # sanity check
        assert rays.shape[-1] == 3, f"Unexpected ray output shape: {rays.shape}"

        return rays

