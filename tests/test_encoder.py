import torch
import argparse

import sys
from pathlib import Path

root_path = Path(__file__).parent.parent
sys.path.append(str(root_path))

from models.encoder_vit import ViTEncoder, DUST3R_ENCODER

torch.serialization.add_safe_globals([argparse.Namespace])

ckpt_path = Path.cwd().joinpath("checkpoints/pretrained/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth")

batch_size = 2
img_size = 128
channels = 3


def test_weights_actually_change():
    """The load must move the parameters. Random init vs loaded must differ."""
    random_encoder = ViTEncoder(checkpoint_path=None, img_size=img_size, **DUST3R_ENCODER)
    loaded_encoder = ViTEncoder(checkpoint_path=ckpt_path, img_size=img_size, **DUST3R_ENCODER)

    random_state = random_encoder.vit.state_dict()
    loaded_state = loaded_encoder.vit.state_dict()

    assert random_state.keys() == loaded_state.keys()
    unchanged = [k for k in loaded_state if torch.equal(random_state[k], loaded_state[k])]
    assert not unchanged, f"{len(unchanged)} tensors untouched by the load: {unchanged[:5]}"

    return loaded_encoder


def test_every_checkpoint_tensor_is_consumed():
    """All 292 encoder tensors must land - not a subset, not zero."""
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    encoder_keys = [
        k for k in checkpoint["model"]
        if k.startswith(("patch_embed.", "enc_blocks.", "enc_norm."))
    ]
    assert len(encoder_keys) == 292, f"expected 292 encoder tensors, found {len(encoder_keys)}"

    # ViTEncoder loads strict=True, so a successful construction already proves
    # the names and shapes line up exactly. This asserts the count is unchanged.
    encoder = ViTEncoder(checkpoint_path=ckpt_path, img_size=img_size, **DUST3R_ENCODER)
    assert len(encoder.vit.state_dict()) == len(encoder_keys)


def test_mismatched_dims_raise():
    """A wrong-width encoder must fail loudly, not silently skip the weights."""
    try:
        ViTEncoder(checkpoint_path=ckpt_path, img_size=img_size, patch_size=16, embed_dim=128)
    except RuntimeError:
        return
    raise AssertionError("embed_dim=128 accepted a 1024-wide checkpoint without raising")


def test_forward_shape_and_freezing():
    encoder = ViTEncoder(checkpoint_path=ckpt_path, img_size=img_size, **DUST3R_ENCODER)
    outputs = encoder(torch.randn(batch_size, channels, img_size, img_size))

    patches = (img_size // DUST3R_ENCODER["patch_size"]) ** 2  # 128/16 -> 8x8 = 64
    assert outputs["tokens"].shape == (batch_size, patches, DUST3R_ENCODER["embed_dim"])

    assert not any(p.requires_grad for p in encoder.vit.parameters()), "encoder should be frozen"


if __name__ == "__main__":
    test_weights_actually_change()
    test_every_checkpoint_tensor_is_consumed()
    test_mismatched_dims_raise()
    test_forward_shape_and_freezing()
    print("encoder checks passed")
