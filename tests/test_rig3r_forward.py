# tests/test_rig3r_forward.py
import torch
from pathlib import Path

import sys

root_path = Path(__file__).parent.parent
sys.path.append(str(root_path))

from models.rig3r import Rig3R

def test_rig3r_forward():
    B, N, C, H, W = 2, 3, 3, 128, 128  # batch, views, channels, image size
    dummy_images = torch.randn(B, N, C, H, W)

    # Optional: add slight variation per view to mimic multi-view captures
    for b in range(B):
        for n in range(1, N):
            dummy_images[b, n] = dummy_images[b, 0] + 0.05*torch.randn_like(dummy_images[b, 0])
            dummy_images[b, n] = dummy_images[b, n].clamp(0.0, 1.0)  # ensure valid RGB

    # Optional dummy metadata
    metadata = {
        "cam2rig": torch.eye(4).repeat(B, N, 1, 1)  # (B, V, 4, 4) per-view SE(3)
    }

    ckpt_path = Path.cwd().joinpath("checkpoints/pretrained/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth")

    # encoder dims are fixed by the DUSt3R ViT-L/16 checkpoint; only the decoder
    # is free to be small here
    model = Rig3R(
        encoder_ckpt=ckpt_path,
        img_size=H,
        patch_size=16,
        embed_dim=1024,
        num_decoder_layers=1,  # small for test
        num_heads=8,
        mlp_dim=1024
    )

    # Forward pass
    with torch.no_grad():
        outputs = model(dummy_images, metadata)

    # Check output keys
    assert "pointmap" in outputs
    assert "pointmap_conf" in outputs  # DPT head also outputs confidence
    assert "pose_raymap" in outputs
    assert "rig_raymap" in outputs

    # Print output shapes
    print("pointmap:", outputs["pointmap"].shape)  # (B, V, H*W, 3) dense predictions
    print("pointmap_conf:", outputs["pointmap_conf"].shape)  # (B, V, H*W, 1) confidence
    print("pose_raymap:", outputs["pose_raymap"].shape)
    print("rig_raymap:", outputs["rig_raymap"].shape)

    # Basic shape checks
    B, V, N_pixels, C_embed = outputs["pointmap"].shape  # N_pixels = H*W (dense)
    assert B == 2
    assert C_embed == 3  # 3D pointmap
    assert outputs["pointmap_conf"].shape[-1] == 1  # confidence is 1D
    print("Forward pass test passed!")

def test_view_fold_matches_per_view_loop():
   """Folding views into the batch dim must be a pure speedup, not a change in math.


   Guards models/rig3r.py: the encoder runs on B*N images at once instead of N
   calls of B images, so token order after the reshape has to match what looping
   and concatenating used to produce.
   """
   B, N, H = 2, 3, 64
   torch.manual_seed(0)
   images = torch.randn(B, N, 3, H, H)
   metadata = {"cam2rig": torch.eye(4).repeat(B, N, 1, 1)}


   model = Rig3R(
       encoder_ckpt=None,
       img_size=H,
       patch_size=8,
       embed_dim=64,
       num_decoder_layers=1,
       num_heads=2,
       mlp_dim=128,
   )
   model.eval()


   with torch.no_grad():
       folded = model(images, metadata)


       # the old path: encode one view at a time, concatenate along the token axis
       looped_tokens = torch.cat(
           [model.encoder(images[:, i])["tokens"] for i in range(N)], dim=1
       )
       expected = model.decoder(
           looped_tokens, frames=N, metadata=metadata, cam2rig=metadata["cam2rig"]
       )


   print("looped_tokens:", looped_tokens.shape, "keys:", sorted(folded.keys()))

   assert folded.keys() == expected.keys()
   for key in expected:
       torch.testing.assert_close(folded[key], expected[key], rtol=1e-4, atol=1e-5)
       print(f"{key}: {tuple(folded[key].shape)} max abs diff "
             f"{(folded[key] - expected[key]).abs().max().item():.3e}")


if __name__ == "__main__":
    test_rig3r_forward()
    test_view_fold_matches_per_view_loop()