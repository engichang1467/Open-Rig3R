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
    

# class SmallHead(nn.Module):
#     """
#         Small MLP head applied per token
#     """
#     def __init__(self, embed_dim, hidden, out_dim):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(embed_dim, hidden),
#             nn.GELU(),
#             nn.Linear(hidden, out_dim)
#         )
    
#     def forward(self, tokens): # tokens: (B, T, C)
#         return self.net(tokens) # (B, T, out_dim)
    

class RigAwareTransformerDecoder(nn.Module):
    """
        Joint self-attention decoder that attends over concatenated patch tokens from
        all frames + optional metadata tokens.

        Inputs:
        - tokens: Tensor (B, V * P, C) where V = frames/views, P = patches per view
        - frames: int, number of frames/views per example (V)
        - metadata: Optional[Tensor] (B, M, meta_dim) or None

        Outputs:
        Dict with:
            - pointmap: (B, V, P, 3)
            - pose_raymap: (B, V, P, 3)
            - rig_raymap: (B, V, P, 3)
    """
    def __init__(
            self,
            embed_dim = 1024,
            num_layers = 8,
            num_heads = 8,
            mlp_dim = 4096,
            metadata_dim = 64,
            metadata_tokens = 1,
            metadata_dropout = 0.5,
            head_hidden = None,
            attn_dropout = 0.0 
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.metadata_dim = metadata_dim
        self.metadata_tokens = metadata_tokens

        # --- Per-key projections ---
        self.key_projs = nn.ModuleDict({
            "cam2rig": nn.Linear(3, embed_dim)  # 3 tokens, each 3D
        })

        # metadata projector: maps external metadata -> embed_dim tokens
        self.meta_proj = nn.Linear(metadata_dim, embed_dim) if metadata_dim is not None else None
        # if metadata is not provided, we still support learned metadata tokens (learnable)
        self.learned_meta = nn.Parameter(torch.randn(1, metadata_tokens, embed_dim))

        self.metadata_dropout = nn.Dropout(metadata_dropout)

        # transformer layers
        self.layers = nn.ModuleList([
            PreNormTransformerBlock(embed_dim, num_heads, mlp_dim, dropout=attn_dropout)
            for _ in range(num_layers)
        ])

        # # Heads: small MLPs to predict vector fields per token
        # if head_hidden is None:
        #     head_hidden = max(embed_dim // 4, 128)
        # self.pointmap_head = SmallHead(embed_dim, head_hidden, 3)
        # self.pose_raymap_head = SmallHead(embed_dim, head_hidden, 3)
        # self.rig_raymap_head = SmallHead(embed_dim, head_hidden, 3)

        self.pointmap_head = PointMapHead(in_dim=embed_dim)
        self.pose_raymap_head = PoseRaymapHead(in_dim=embed_dim)
        self.rig_raymap_head = RigRaymapHead(in_dim=embed_dim)

        # optional final normalization before heads (stable)
        self.final_ln = nn.LayerNorm(embed_dim)
    
    def _collect_metadata_tensors(self, metadata, device):
        """
        Extracts and normalizes metadata dict entries into a list of (B, M, D) tensors.
        """
        meta_list = []
        for key, value in metadata.items():
            if value is None:
                continue
            v = value.to(device)
            if v.dim() == 2:
                v = v.unsqueeze(1) # (B, D) -> (B, 1, D)
            elif v.dim() != 3:
                raise ValueError(f"Unsupported metadata shape for key {key}: {v.shape}")
            # meta_list.append(v)
            
            # Project each token using per-key projection
            # proj_layer = self.key_projs.get(key)
            proj_layer = self.key_projs[key]
            if proj_layer is None:
                raise ValueError(f"No projection defined for metadata key: {key}")
            # Flatten tokens along sequence for linear, then reshape back
            B, M, D = v.shape
            v_flat = v.reshape(B * M, D)
            v_proj = proj_layer(v_flat).reshape(B, M, self.embed_dim)
            
            meta_list.append(v_proj)
            
        return meta_list
    
    def _prepare_metadata_tokens(self, metadata, batch_size, device):
        """
            Returns metadata tokens of shape (B, M, C).
            - If metadata is provided: project it -> (B, M, C).
            Expect metadata shape (B, M, metadata_dim). If metadata has fewer tokens,
            it's still fine (we project element-wise).
            - If None: use learned tokens repeated over batch.
        """
        if metadata is None:
            meta_tokens = self.learned_meta.expand(batch_size, -1, -1).to(device)  # (B, M, C)
        
        meta_list = self._collect_metadata_tensors(metadata, device)

        if len(meta_list) == 0:
            return self.learned_meta.expand(batch_size, -1, -1).to(device)

        meta_tokens = torch.cat(meta_list, dim=1)

        if self.meta_proj:
            
        # if self.meta_proj is None:
        #     raise ValueError("Decoder expects metadata_dim but meta_proj is None")

            meta_tokens = self.meta_proj(meta_tokens)
        
        return self.metadata_dropout(meta_tokens)
    
         # else:
        #     # metadata: (B, M, metadata_dim) -> project to embed_dim
        #     if self.meta_proj is None:
        #         raise ValueError("Decoder expects metadata_dim but meta_proj is None")
            
        #     meta_tokens = self.meta_proj(metadata.to(device))  # (B, M, C)
        # meta_tokens = self.metadata_dropout(meta_tokens)
        # return meta_tokens
    
    def forward(self, tokens, frames, metadata=None, cam2rig=None):
        """
            tokens: (B, V * P, C)
            frames: V (int)
            metadata: Optional (B, M, metadata_dim)  (M == metadata_tokens recommended)
        """
        B, T_total, C = tokens.shape
        # print(f"T_total: {T_total}")
        # print(f"frames: {frames}")
        assert C == self.embed_dim, f"tokens embed dim {C} != decoder embed_dim {self.embed_dim}"
        assert T_total % frames == 0, f"tokens length must be divisible by frames"
        patches_per_frame = T_total // frames

        device = tokens.device

        # print(f"device: {device}")

        # prepare metadata tokens (B, M, C)
        meta_tokens = self._prepare_metadata_tokens(metadata, B, device)  # (B, M, C)

        # concatenate metadata tokens in front of patch tokens: (B, M + T_total, C)
        seq = torch.cat([meta_tokens, tokens], dim=1)

        # run joint transformer layers
        for layer in self.layers:
            seq = layer(seq)

        # extract processed patch tokens (skip metadata tokens)
        proc_patches = seq[:, meta_tokens.shape[1]:, :]  # (B, T_total, C)
        proc_patches = self.final_ln(proc_patches)

        # reshape into (B, V, P, C)
        proc_patches = proc_patches.view(B, frames, patches_per_frame, C)

        # apply heads per token
        # flatten tokens for head MLPs then reshape back
        flat = proc_patches.reshape(B * frames * patches_per_frame, C)  # (B*V*P, C)
        point_preds = self.pointmap_head(flat).reshape(B, frames, patches_per_frame, 3)
        pose_preds = self.pose_raymap_head(flat).reshape(B, frames, patches_per_frame, 3)
        # rig_preds = self.rig_raymap_head(flat, cam2rig=cam2rig).reshape(B, frames, patches_per_frame, 3)
        
        N = frames * patches_per_frame
        flat_reshaped = flat.view(B, N, C)
        rig_preds = self.rig_raymap_head(flat_reshaped, cam2rig=cam2rig)

        # print(f"[DEBUG] rig_preds.shape={rig_preds.shape}, "
        #       f"B={B}, frames={frames}, patches_per_frame={patches_per_frame}, C={C}, "
        #       f"expected={B*frames*patches_per_frame*3}")

        # print(f"[DEBUG] rig_preds.shape={rig_preds.shape}, expected=({B},{frames*patches_per_frame},3)")

        rig_preds = rig_preds.view(B, frames, patches_per_frame, 3)

        # print(rig_preds.mean(), rig_preds.std())

        # print(f"rig_preds stats:\nmean: {rig_preds.mean()}\nstd: {rig_preds.std()}\n")
 
        return {
            "pointmap": point_preds,
            "pose_raymap": pose_preds,
            "rig_raymap": rig_preds,
            "features": proc_patches  # (B, V, P, C) for debugging / downstream heads if needed
        }