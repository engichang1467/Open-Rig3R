import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from models.heads.pointmap_head import PointMapHead 
from models.heads.rig_raymap_head import RigRaymapHead
from models.heads.pose_raymap_head import PoseRaymapHead

class PreNormTransformerBlock(nn.Module):
    """
        Pre-norm transformer block: LN -> MHA -> resid -> LN -> MLP -> resid
    """
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        # x: (B, S, C)
        x_ln = self.ln1(x)
        attn_out, _ = self.mha(x_ln, x_ln, x_ln, need_weights=False)
        x = x + self.dropout(attn_out)
        x = x + self.mlp(self.ln2(x))
        return x
    

def sincos1d(values, dim, max_period=10000.0):
    """1D sine-cosine embedding of arbitrary values. (..., ) -> (..., dim)

    Takes floats, not just indices, so the same encoding covers the discrete IDs
    and the normalized timestamp (Rig3R sec 3.3).
    """
    assert dim % 2 == 0, f"sincos1d needs an even dim, got {dim}"
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=values.device) / half
    )
    args = values.float().unsqueeze(-1) * freqs
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


# Rig3R sec 3.3 metadata fields, in the order their slices are concatenated.
# The rig raymap patch is per-patch rather than per-view, so it is built separately.
METADATA_FIELDS = ("frame_index", "camera_id", "timestamp", "rig_raymap")


class RigAwareTransformerDecoder(nn.Module):
    """
        Joint self-attention decoder over concatenated patch tokens from all frames,
        with rig-aware metadata added to each patch token.

        Inputs:
        - tokens: Tensor (B, V * P, C) where V = frames/views, P = patches per view
        - frames: int, number of frames/views per example (V)
        - metadata: Optional dict of per-view tensors; see _metadata_embedding

        Outputs:
        Dict with:
            - pointmap: (B, V, H*W, 3) dense pixel-level predictions
            - pointmap_conf: (B, V, H*W, 1) per-pixel confidence
            - pose_raymap: (B, V, P, 3)
            - rig_raymap: (B, V, P, 6)
    """
    def __init__(
            self,
            embed_dim = 1024,
            num_layers = 8,
            num_heads = 8,
            mlp_dim = 4096,
            metadata_dropout = 0.5,
            head_hidden = None,
            attn_dropout = 0.0,
            img_size = 384,
            patch_size = 8
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.img_size = img_size
        self.patch_size = patch_size
        self.metadata_dropout = metadata_dropout

        # sec 3.3: the metadata components are concatenated and added to the patch
        # tokens, so each one owns an equal slice of the embedding
        assert embed_dim % (2 * len(METADATA_FIELDS)) == 0, (
            f"embed_dim {embed_dim} must divide into {len(METADATA_FIELDS)} even slices"
        )
        self.meta_dim = embed_dim // len(METADATA_FIELDS)

        # transformer layers
        self.layers = nn.ModuleList([
            PreNormTransformerBlock(embed_dim, num_heads, mlp_dim, dropout=attn_dropout)
            for _ in range(num_layers)
        ])

        self.pointmap_head = PointMapHead(
            in_dim=embed_dim,
            hidden_dim=256,
            img_size=img_size,
            patch_size=patch_size
        )
        self.pose_raymap_head = PoseRaymapHead(in_dim=embed_dim)
        self.rig_raymap_head = RigRaymapHead(in_dim=embed_dim)

        # optional final normalization before heads (stable)
        self.final_ln = nn.LayerNorm(embed_dim)
    
    def _keep_mask(self, B, dims, device):
        """Per-sample keep/drop mask for one metadata field. (B, 1, ... ) broadcastable

        Sec 3.4 Embedding Dropout: each field is dropped with 50% probability during
        training so the model learns to infer missing context rather than lean on it.
        Masking the *embedding* rather than the raw value is what makes a dropped field
        read as absent - a zeroed camera ID would still encode as camera zero.

        Not scaled by 1/(1-p) the way nn.Dropout is: absent has to look at inference
        time exactly like it did during training, or dropout itself becomes a train
        /test mismatch.
        """
        if not self.training or self.metadata_dropout <= 0:
            return torch.ones((1,) * dims, device=device)
        keep = torch.rand(B, device=device) >= self.metadata_dropout
        return keep.float().view(B, *([1] * (dims - 1)))

    def _metadata_embedding(self, metadata, B, frames, patches_per_frame, device):
        """Per-patch metadata embedding to add to the patch tokens. (B, V * P, C)

        Each field owns an equal slice of the embedding. A field that is absent or
        dropped leaves its slice at zero, which is the whole masking mechanism.

        Expected shapes, all optional except the frame index:
            frame_index: (B, V) long   - sec 3.3 says N is always included, so it is
                                         synthesised from view order when missing
            camera_id:   (B, V) long
            timestamp:   (B, V) float, seconds
            rig_raymap:  (B, V, P, 6)  - per-patch, not yet wired in
        """
        metadata = metadata or {}

        frame_index = metadata.get("frame_index")
        if frame_index is None:
            frame_index = torch.arange(frames, device=device).expand(B, frames)

        # per-view fields: encoded once per view, then broadcast across its patches.
        # The frame index is never dropped (sec 3.3), the rest are.
        per_view = [sincos1d(frame_index.to(device), self.meta_dim)]
        for key in ("camera_id", "timestamp"):
            value = metadata.get(key)
            if value is None:
                per_view.append(torch.zeros(B, frames, self.meta_dim, device=device))
            else:
                encoded = sincos1d(value.to(device), self.meta_dim)
                per_view.append(encoded * self._keep_mask(B, encoded.dim(), device))

        embedding = torch.cat(per_view, dim=-1).unsqueeze(2)  # (B, V, 1, 3 * meta_dim)
        embedding = embedding.expand(B, frames, patches_per_frame, -1)

        # The rig raymap patch is genuinely per-patch rather than per-view. Wiring it
        # in is gated on metadata dropout landing first: it is also the rig raymap
        # head's target, so without masking it hands the answer straight to the head.
        assert metadata.get("rig_raymap") is None, (
            "rig raymap metadata needs per-field dropout (#39) before it can be used"
        )
        rig_slice = torch.zeros(B, frames, patches_per_frame, self.meta_dim, device=device)

        embedding = torch.cat([embedding, rig_slice], dim=-1)
        return embedding.reshape(B, frames * patches_per_frame, self.embed_dim)

    def forward(self, tokens, frames, metadata=None):
        """
            tokens: (B, V * P, C)
            frames: V (int)
            metadata: Optional dict of per-view tensors; see _metadata_embedding
        """
        B, T_total, C = tokens.shape
        assert C == self.embed_dim, f"tokens embed dim {C} != decoder embed_dim {self.embed_dim}"
        assert T_total % frames == 0, f"tokens length must be divisible by frames"
        patches_per_frame = T_total // frames

        device = tokens.device

        # sec 3.3: metadata is added to every patch token rather than attended to as
        # separate tokens, so each patch carries its own copy through to the heads
        seq = tokens + self._metadata_embedding(
            metadata, B, frames, patches_per_frame, device
        )

        # run joint transformer layers
        for layer in self.layers:
            seq = layer(seq)

        proc_patches = self.final_ln(seq)  # (B, T_total, C)

        # reshape into (B, V, P, C)
        proc_patches = proc_patches.view(B, frames, patches_per_frame, C)

        # apply heads per token
        # flatten tokens for head MLPs then reshape back
        flat = proc_patches.reshape(B * frames * patches_per_frame, C)  # (B*V*P, C)

        # DPT pointmap head: expects (B*V, P, C), returns (B*V, H*W, 3), (B*V, H*W, 1)
        dpt_input = proc_patches.reshape(B * frames, patches_per_frame, C)  # (B*V, P, C)
        point_preds, conf_preds = self.pointmap_head(dpt_input)
        # Reshape to (B, V, H*W, 3) and (B, V, H*W, 1)
        H_W = point_preds.shape[1]  # H*W = img_size * img_size
        point_preds = point_preds.view(B, frames, H_W, 3)
        conf_preds = conf_preds.view(B, frames, H_W, 1)

        pose_preds = self.pose_raymap_head(flat).reshape(B, frames, patches_per_frame, 3)

        N = frames * patches_per_frame
        flat_reshaped = flat.view(B, N, C)
        rig_preds = self.rig_raymap_head(flat_reshaped)

        rig_preds = rig_preds.view(B, frames, patches_per_frame, 6)

        return {
            "pointmap": point_preds,  # (B, V, H*W, 3) dense predictions
            "pointmap_conf": conf_preds,  # (B, V, H*W, 1) confidence
            "pose_raymap": pose_preds,
            "rig_raymap": rig_preds,
            "features": proc_patches  # (B, V, P, C) for debugging / downstream heads if needed
        }