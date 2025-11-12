import torch

import sys
from pathlib import Path

root_path = Path(__file__).parent.parent
sys.path.append(str(root_path))

from models.encoder_vit import ViTEncoder

ckpt_path = Path.cwd().joinpath("checkpoints/pretrained/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth")

# Parameters
batch_size = 2
img_size = 384
patch_size = 8
channels = 3

# Create dummy input
dummy_input = torch.randn(batch_size, channels, img_size, img_size)

# Initialize encoder (no checkpoint to keep it simple)
# encoder = ViTEncoder(checkpoint_path=None, img_size=img_size, patch_size=patch_size)
encoder = ViTEncoder(checkpoint_path=ckpt_path, img_size=img_size, patch_size=patch_size)


# Forward pass
outputs = encoder(dummy_input)

print("Patch tokens shape:", outputs["tokens"].shape)      # (B, N, C)
print("Feature map shape:", outputs["feature_map"].shape)  # (B, C, H, W)

# Check if pos_embed is frozen
pos_embed_requires_grad = encoder.vit.pos_embed.requires_grad
print("Positional embedding requires_grad:", pos_embed_requires_grad)
