# tests/test_rig3r_forward.py
import torch
from pathlib import Path

import sys

root_path = Path(__file__).parent.parent
sys.path.append(str(root_path))

from models.rig3r import Rig3R

def test_rig3r_forward():
    B, N, C, H, W = 2, 3, 3, 384, 384  # batch, views, channels, image size
    # B, N, C, H, W = 1, 1, 3, 128, 128  # batch, views, channels, image size
    dummy_images = torch.randn(B, N, C, H, W)

    # Optional: add slight variation per view to mimic multi-view captures
    for b in range(B):
        for n in range(1, N):
            dummy_images[b, n] = dummy_images[b, 0] + 0.05*torch.randn_like(dummy_images[b, 0])
            dummy_images[b, n] = dummy_images[b, n].clamp(0.0, 1.0)  # ensure valid RGB

    # Optional dummy metadata
    metadata = {
        "cam2rig": torch.eye(3).unsqueeze(0).repeat(B, 1, 1)  # (B, 3, 3)
        # "cam2rig": torch.eye(3).unsqueeze(0).repeat(B, 3)  # (B, 3, 3)
    }

    # Initialize model
    # model = Rig3R(
    #     encoder_ckpt=None,    # use sinusoidal encoder for test
    #     img_size=H,
    #     patch_size=16,
    #     embed_dim=64,
    #     metadata_dim=64,
    #     num_decoder_layers=1,  # small for test
    #     num_heads=2,
    #     mlp_dim=128
    # )

    ckpt_path = Path.cwd().joinpath("checkpoints/pretrained/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth")

    model = Rig3R(
        encoder_ckpt=ckpt_path,    # use sinusoidal encoder for test
        # encoder_ckpt=None,    # use sinusoidal encoder for test
        img_size=H,
        patch_size=8,
        embed_dim=384,
        metadata_dim=384,
        num_decoder_layers=4,  # small for test
        num_heads=6,
        mlp_dim=384*4
    )

    # Forward pass
    with torch.no_grad():
        outputs = model(dummy_images, metadata)

    # Check output keys
    assert "pointmap" in outputs
    assert "pose_raymap" in outputs
    assert "rig_raymap" in outputs

    # Print output shapes
    print("pointmap:", outputs["pointmap"].shape)
    print("pose_raymap:", outputs["pose_raymap"].shape)
    print("rig_raymap:", outputs["rig_raymap"].shape)

    # Basic shape checks
    B, V, N_patches, C_embed = outputs["pointmap"].shape
    assert B == 2
    assert C_embed == 3  # 3D pointmap / 3D rays
    print("Forward pass test passed!")

if __name__ == "__main__":
    test_rig3r_forward()
