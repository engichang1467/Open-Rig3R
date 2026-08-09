# tests/test_rig3r_forward.py
import inspect

import torch
from pathlib import Path

import sys

root_path = Path(__file__).parent.parent
sys.path.append(str(root_path))

from models.rig3r import Rig3R
from models.decoder_transformer import RigAwareTransformerDecoder
from models.heads.rig_raymap_head import RigRaymapHead
from utils.raymap import build_raymap_targets

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
           looped_tokens, frames=N, metadata=metadata
       )


   print("looped_tokens:", looped_tokens.shape, "keys:", sorted(folded.keys()))

   assert folded.keys() == expected.keys()
   for key in expected:
       torch.testing.assert_close(folded[key], expected[key], rtol=1e-4, atol=1e-5)
       print(f"{key}: {tuple(folded[key].shape)} max abs diff "
             f"{(folded[key] - expected[key]).abs().max().item():.3e}")


# --- rig raymap head must not see ground-truth extrinsics -------------------
# The rig raymap target's origin half is exactly cam2rig's translation
# (utils/raymap.py builds it as translation.expand_as), so a head handed the same
# matrix reproduces that half for free. Rig pose reaches the model only through
# the metadata embedding.

LEAK_IMAGE_SIZE = (64, 64)
LEAK_PATCH_SIZE = 16
LEAK_PATCHES = (LEAK_IMAGE_SIZE[0] // LEAK_PATCH_SIZE) ** 2
LEAK_VIEWS = 2
LEAK_MOUNT = torch.tensor([1.5, -0.5, 0.25])  # view 1 mounted off the rig origin


def _rig_target():
    """Rig raymap target for a 2-view rig, view 1 physically offset from view 0."""
    cam2rig = torch.eye(4).repeat(1, LEAK_VIEWS, 1, 1)
    cam2rig[0, 1, :3, 3] = LEAK_MOUNT
    intrinsics = torch.tensor([[[32.0, 32.0, 32.0, 32.0]] * LEAK_VIEWS])
    world_from_rig = torch.eye(4).repeat(1, LEAK_VIEWS, 1, 1)
    targets = build_raymap_targets(
        cam2rig, intrinsics, world_from_rig, LEAK_IMAGE_SIZE, LEAK_PATCH_SIZE
    )
    return cam2rig, targets["rig_raymap"]  # (1, V, P, 6)


def _leaking_transform(rays, cam2rig):
    """The deleted `R @ pred + t` branch, kept here to measure against."""
    B, views, _, _ = cam2rig.shape
    N = rays.shape[1]
    origins, directions = rays[..., :3], rays[..., 3:]
    R, t = cam2rig[..., :3, :3], cam2rig[..., :3, 3]
    origins = origins.view(B, views, N // views, 3)
    directions = directions.view(B, views, N // views, 3)
    origins = torch.einsum("bvij,bvnj->bvni", R, origins) + t.unsqueeze(2)
    directions = torch.einsum("bvij,bvnj->bvni", R, directions)
    return torch.cat([origins.reshape(B, N, 3), directions.reshape(B, N, 3)], dim=-1)


def test_zero_prediction_no_longer_matches_rig_target():
    """A head predicting nothing must score badly, not perfectly."""
    cam2rig, target = _rig_target()
    target_origins = target[..., :3].reshape(1, LEAK_VIEWS * LEAK_PATCHES, 3)
    zeros = torch.zeros(1, LEAK_VIEWS * LEAK_PATCHES, 6)

    leaked_error = (_leaking_transform(zeros, cam2rig)[..., :3] - target_origins).abs().max()
    honest_error = (zeros[..., :3] - target_origins).abs().max()

    print(f"leaking path, zero prediction -> origin error {leaked_error:.3e}")
    print(f"current path, zero prediction -> origin error {honest_error:.3e}")

    assert leaked_error < 1e-6, "expected the old path to reproduce origins exactly"
    assert honest_error > 1.0, "zero prediction must not satisfy the rig target"
    print("Zero prediction no longer satisfies the rig target test passed!")


def test_extrinsics_never_reach_rig_head():
    """No route for cam2rig into the head, positionally or by keyword."""
    params = inspect.signature(RigAwareTransformerDecoder.forward).parameters
    assert "cam2rig" not in params, f"decoder.forward takes cam2rig: {list(params)}"

    head = RigRaymapHead(in_dim=32, hidden_dim=16).eval()
    tokens = torch.randn(1, LEAK_VIEWS * LEAK_PATCHES, 32)
    cam2rig, _ = _rig_target()

    try:
        head(tokens, cam2rig)
    except TypeError:
        pass
    else:
        raise AssertionError("rig raymap head still accepts a second argument")
    print("No extrinsics reach the rig raymap head test passed!")


def test_normalize_touches_directions_only():
    """normalize=True is off by default, so nothing else covers this branch."""
    torch.manual_seed(0)
    head = RigRaymapHead(in_dim=32, hidden_dim=16, normalize=True).eval()
    plain = RigRaymapHead(in_dim=32, hidden_dim=16, normalize=False).eval()
    plain.load_state_dict(head.state_dict())

    tokens = torch.randn(1, LEAK_VIEWS * LEAK_PATCHES, 32)
    with torch.no_grad():
        normed, raw = head(tokens), plain(tokens)

    torch.testing.assert_close(normed[..., :3], raw[..., :3])  # origins untouched
    norms = normed[..., 3:].norm(dim=-1)
    torch.testing.assert_close(norms, torch.ones_like(norms))
    print(f"Origins preserved, direction norms in "
          f"[{norms.min():.6f}, {norms.max():.6f}] test passed!")


if __name__ == "__main__":
    test_rig3r_forward()
    test_view_fold_matches_per_view_loop()
    test_zero_prediction_no_longer_matches_rig_target()
    test_extrinsics_never_reach_rig_head()
    test_normalize_touches_directions_only()