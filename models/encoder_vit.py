import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.vision_transformer import VisionTransformer

# allow argparse.Namespace inside the checkpoint
torch.serialization.add_safe_globals([argparse.Namespace])

class ViTEncoder(nn.Module):
    def __init__(self, checkpoint_path=None, img_size=384, patch_size=8, embed_dim=1024):
        super().__init__()
        # Load pretrained ViT-Large architecture
        self.vit = VisionTransformer(
            image_size=img_size,
            patch_size=patch_size,
            num_layers=24,
            num_heads=8,
            hidden_dim=embed_dim,
            mlp_dim=4096,
            num_classes=0
        )

        # Replace positional embedding with sinusoidal (frozen)
        num_patches = self.vit.dump_patches
        self.vit.pos_embed = nn.Parameter(
            self._build_sinusoidal_pos_embed(num_patches, embed_dim), requires_grad=False
        )

        # Load DUST3R weights
        # Optional: load DUSt3R weights for other parameters if provided
        if checkpoint_path is not None:
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            self._load_dust3r_weights(state_dict)

        # LayerNorm for output normalization
        self.norm = nn.LayerNorm(embed_dim)

    def _build_sinusoidal_pos_embed(self, num_patches, dim):
        # Standard 2D sinusoidal positional encoding
        H = W = int(num_patches ** 0.5)
        grid_y, grid_x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        grid = torch.stack([grid_x, grid_y], dim=-1).float() # shape (H, W, 2)

        pos_embed = torch.zeros(H, W, dim)
        div_term = torch.exp(torch.arange(0, dim, 4).float() * -(torch.log(torch.tensor(10000.0)) / dim))
        pos_embed[:, :, 0::4] = torch.sin(grid[:, :, 0:1] * div_term)
        pos_embed[:, :, 1::4] = torch.cos(grid[:, :, 0:1] * div_term)
        pos_embed[:, :, 2::4] = torch.sin(grid[:, :, 1:2] * div_term)
        pos_embed[:, :, 3::4] = torch.cos(grid[:, :, 1:2] * div_term)

        pos_embed = pos_embed.reshape(1, H * W, dim)
        cls_token = torch.zeros(1, 1, dim) # [CLS] token
        return torch.cat([cls_token, pos_embed], dim=1)

    def _embed_patches(self, x):
        B = x.shape[0]

        # Patch embedding
        tokens = self.vit.conv_proj(x) # (B, N, C)
        # Flatten spatial dims and transpose to (B, N, C)
        B, C, H_p, W_p = tokens.shape
        tokens = tokens.flatten(2).transpose(1, 2)
        # Prepend [CLS] token
        cls_token = self.vit.class_token.expand(B, -1, -1) # (B, 1, C)
        tokens = torch.cat((cls_token, tokens), dim=1) # (B, N+1, C)

        # Add positional embedding
        tokens = tokens + self.vit.pos_embed.to(tokens.device)
        return tokens

    def _tokens_to_feature_map(self, patch_tokens):
        # Reshape tokens into a spatial feature map
        B, N, C = patch_tokens.shape
        H = W = int(N ** 0.5)
        return patch_tokens.transpose(1, 2).reshape(B, C, H, W)
    
    def _load_dust3r_weights(self, state_dict):
        # Load pretrained weights for parameters other than pos_embed
        state_dict.pop("pos_embed", None) # Remove pos_embed to keep sinusoidal
        msg = self.vit.load_state_dict(state_dict, strict=False)
        print("Loaded DUSt3R weights (excluding pos_embed)")
        # print("Loaded DUSt3R weights (excluding pos_embed):", msg)

    def forward(self, x):

        tokens = self._embed_patches(x)
        tokens = self.vit.encoder(tokens)
        patch_tokens = tokens[:, 1:, :]
        feature_map = self._tokens_to_feature_map(patch_tokens)
        
        return {
            "tokens": self.norm(patch_tokens),
            "feature_map": feature_map
        }
