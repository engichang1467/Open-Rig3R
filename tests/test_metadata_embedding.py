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


def build_decoder(metadata_dropout=0.0):
    return RigAwareTransformerDecoder(
        embed_dim=C,
        num_layers=1,
        num_heads=2,
        mlp_dim=C * 2,
        metadata_dropout=metadata_dropout,
        attn_dropout=0.0,
        img_size=IMG_SIZE,
        patch_size=PATCH_SIZE,
    ).eval()


def full_metadata(B_=B, V_=V, rig=False):
    metadata = {
        "frame_index": torch.arange(V_).expand(B_, V_),
        "camera_id": torch.arange(V_).expand(B_, V_) % 2,
        "timestamp": torch.linspace(0, 1, V_).expand(B_, V_),
    }
    if rig:
        metadata["rig_raymap"] = torch.randn(B_, V_, P, 6)
    return metadata


def slice_of(embedding, field):
    """Pull one field's slice out of a (B, V, P, C) metadata embedding."""
    index = METADATA_FIELDS.index(field)
    width = C // len(METADATA_FIELDS)
    return embedding[..., index * width:(index + 1) * width]


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


def test_dropout_is_off_at_eval():
    """Inference must see every field it was given, every time."""
    decoder = build_decoder(metadata_dropout=0.5).eval()
    metadata = full_metadata()

    for _ in range(20):
        embedding = decoder._metadata_embedding(
            metadata, B, V, P, torch.device("cpu")
        ).view(B, V, P, C)
        assert slice_of(embedding, "camera_id").abs().max() > 0
        assert slice_of(embedding, "timestamp").abs().max() > 0

    print("Dropout is inactive at eval test passed!")


def test_dropout_masks_whole_fields_per_sample():
    """Sec 3.4: each field is dropped independently, per sample, at ~50%."""
    torch.manual_seed(0)
    decoder = build_decoder(metadata_dropout=0.5).train()
    metadata = full_metadata(B_=64)

    dropped = {"camera_id": 0, "timestamp": 0, "frame_index": 0}
    trials = 40
    for _ in range(trials):
        embedding = decoder._metadata_embedding(
            metadata, 64, V, P, torch.device("cpu")
        ).view(64, V, P, C)
        for field in dropped:
            # a dropped sample has that field's slice fully zeroed
            per_sample = slice_of(embedding, field).abs().amax(dim=(1, 2, 3))
            dropped[field] += int((per_sample == 0).sum())

    total = trials * 64
    for field in ("camera_id", "timestamp"):
        rate = dropped[field] / total
        print(f"{field:12s} dropped {rate:.3f} of the time")
        assert 0.4 < rate < 0.6, f"{field} dropped at {rate}, expected ~0.5"

    assert dropped["frame_index"] == 0, "the frame index must never be dropped"
    print("Fields drop independently per sample test passed!")


def test_dropped_field_matches_absent_field():
    """A dropped field must be indistinguishable from one that was never supplied."""
    decoder = build_decoder(metadata_dropout=1.0).train()  # drop everything droppable
    device = torch.device("cpu")

    dropped = decoder._metadata_embedding(full_metadata(), B, V, P, device)
    absent = build_decoder(metadata_dropout=0.0).eval()._metadata_embedding(
        {"frame_index": torch.arange(V).expand(B, V)}, B, V, P, device
    )

    torch.testing.assert_close(dropped, absent)
    print("Dropped reads as absent test passed!")


def test_rig_raymap_varies_per_patch():
    """r_i is a per-patch field; a per-view broadcast would throw its geometry away."""
    decoder = build_decoder()
    metadata = full_metadata(rig=True)

    embedding = decoder._metadata_embedding(
        metadata, B, V, P, torch.device("cpu")
    ).view(B, V, P, C)
    rig = slice_of(embedding, "rig_raymap")

    within_patch_spread = (rig - rig.mean(dim=2, keepdim=True)).abs().max()
    print(f"rig slice spread across patches: {within_patch_spread:.3e}")
    assert within_patch_spread > 1e-3, "rig raymap collapsed to one value per view"

    # the three per-view fields are constant across patches, by contrast
    for field in ("frame_index", "camera_id", "timestamp"):
        chunk = slice_of(embedding, field)
        torch.testing.assert_close(chunk, chunk[:, :, :1].expand_as(chunk))

    print("Rig raymap varies per patch test passed!")


def test_rig_raymap_is_dropped_like_the_other_fields():
    """The head's own target must be withheld half the time, or it just copies."""
    torch.manual_seed(0)
    decoder = build_decoder(metadata_dropout=0.5).train()
    metadata = full_metadata(B_=64, rig=True)

    dropped = 0
    trials = 40
    for _ in range(trials):
        embedding = decoder._metadata_embedding(
            metadata, 64, V, P, torch.device("cpu")
        ).view(64, V, P, C)
        per_sample = slice_of(embedding, "rig_raymap").abs().amax(dim=(1, 2, 3))
        dropped += int((per_sample == 0).sum())

    rate = dropped / (trials * 64)
    print(f"rig_raymap dropped {rate:.3f} of the time")
    assert 0.4 < rate < 0.6, f"rig raymap dropped at {rate}, expected ~0.5"
    print("Rig raymap is dropped like the other fields test passed!")


def test_rig_raymap_reaches_the_tokens():
    """A supplied r_i must change the output, or the projection is dead weight."""
    decoder = build_decoder()
    tokens = torch.randn(B, V * P, C)
    metadata = full_metadata(rig=True)

    with torch.no_grad():
        without = decoder(tokens, frames=V, metadata=full_metadata())["features"]
        with_rig = decoder(tokens, frames=V, metadata=metadata)["features"]

    delta = (with_rig - without).abs().max()
    print(f"feature delta from supplying r_i: {delta:.3e}")
    assert delta > 1e-4, "rig raymap metadata never reached the patch tokens"
    print("Rig raymap reaches the tokens test passed!")


if __name__ == "__main__":
    test_frame_index_distinguishes_identical_views()
    test_frame_index_is_synthesised_when_absent()
    test_absent_field_leaves_its_slice_at_zero()
    test_metadata_is_added_not_prepended()
    test_sincos1d_encodes_continuous_values()
    test_dropout_is_off_at_eval()
    test_dropout_masks_whole_fields_per_sample()
    test_dropped_field_matches_absent_field()
