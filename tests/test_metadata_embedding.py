# tests/test_metadata_embedding.py
"""Rig3R sec 3.3 metadata embedding: added to patch tokens, one slice per field."""
import sys
from pathlib import Path

import torch

root_path = Path(__file__).parent.parent
sys.path.append(str(root_path))

from models.decoder_transformer import METADATA_FIELDS, RigAwareTransformerDecoder, sincos1d

B, V, P, C = 2, 3, 4, 64
IMG_SIZE, PATCH_SIZE = 16, 8


def build_decoder():
    return RigAwareTransformerDecoder(
        embed_dim=C,
        num_layers=1,
        num_heads=2,
        mlp_dim=C * 2,
        attn_dropout=0.0,
        img_size=IMG_SIZE,
        patch_size=PATCH_SIZE,
    ).eval()


def test_frame_index_distinguishes_identical_views():
    """The decoder is permutation-equivariant without the frame index.

    Feed V identical blocks of patch tokens. Attention alone cannot tell them
    apart, so any difference between the per-view outputs comes from N.
    """
    decoder = build_decoder()
    one_view = torch.randn(B, P, C)
    tokens = one_view.repeat(1, V, 1)  # (B, V * P, C), every view identical

    with torch.no_grad():
        features = decoder(tokens, frames=V)["features"]  # (B, V, P, C)

    spread = max(
        (features[:, i] - features[:, j]).abs().max().item()
        for i in range(V)
        for j in range(i + 1, V)
    )
    print(f"max spread across identical views: {spread:.3e}")
    assert spread > 1e-3, "views are indistinguishable - the frame index is not reaching the tokens"
    print("Frame index distinguishes identical views test passed!")


def test_frame_index_is_synthesised_when_absent():
    """Sec 3.3: N is always included, so omitting it must not silently drop it."""
    decoder = build_decoder()
    tokens = torch.randn(B, V * P, C)
    explicit = {"frame_index": torch.arange(V).expand(B, V)}

    with torch.no_grad():
        default = decoder(tokens, frames=V)["features"]
        supplied = decoder(tokens, frames=V, metadata=explicit)["features"]

    torch.testing.assert_close(default, supplied)
    print("Frame index defaults to view order test passed!")


def test_absent_field_leaves_its_slice_at_zero():
    """A dropped field must read as absent, not as some particular value."""
    decoder = build_decoder()
    meta_dim = decoder.meta_dim
    assert meta_dim * len(METADATA_FIELDS) == C

    embedding = decoder._metadata_embedding(
        {"camera_id": torch.zeros(B, V, dtype=torch.long)},
        B, V, P, torch.device("cpu"),
    ).view(B, V, P, C)

    # slice order follows METADATA_FIELDS: frame index, camera ID, timestamp, rig raymap
    for index, field in enumerate(METADATA_FIELDS):
        chunk = embedding[..., index * meta_dim:(index + 1) * meta_dim]
        populated = chunk.abs().max() > 0
        expected = field in ("frame_index", "camera_id")
        assert populated == expected, f"{field} slice populated={populated}, expected {expected}"

    print("Absent fields leave zeroed slices test passed!")


def test_metadata_is_added_not_prepended():
    """Sec 3.3 adds metadata to the patch tokens; the sequence length must not grow."""
    decoder = build_decoder()
    tokens = torch.randn(B, V * P, C)

    with torch.no_grad():
        features = decoder(tokens, frames=V)["features"]

    assert features.shape == (B, V, P, C), features.shape
    assert not hasattr(decoder, "learned_meta"), "prepended metadata tokens still present"
    print("Metadata is added to patch tokens test passed!")


def test_sincos1d_encodes_continuous_values():
    """Timestamps are floats, not indices, so the encoding must separate them."""
    values = torch.tensor([[0.0, 0.05, 1.0]])
    encoded = sincos1d(values, 16)

    assert encoded.shape == (1, 3, 16), encoded.shape
    assert (encoded[0, 0] - encoded[0, 1]).abs().max() > 1e-4, "close timestamps collapsed"
    torch.testing.assert_close(encoded[0, 0, 8:], torch.ones(8))  # cos(0) = 1
    print("sincos1d separates continuous values test passed!")


if __name__ == "__main__":
    test_frame_index_distinguishes_identical_views()
    test_frame_index_is_synthesised_when_absent()
    test_absent_field_leaves_its_slice_at_zero()
    test_metadata_is_added_not_prepended()
    test_sincos1d_encodes_continuous_values()
