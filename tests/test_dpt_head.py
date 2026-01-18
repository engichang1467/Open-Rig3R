import torch
import sys
from pathlib import Path

root_path = Path(__file__).parent.parent
sys.path.append(str(root_path))

from models.heads.pointmap_head import ResidualConvUnit, FusionBlock, PointMapHead


def test_residual_conv_unit():
    """Test ResidualConvUnit preserves shape and adds skip connection."""
    batch_size = 2
    channels = 256
    height, width = 48, 48

    rcu = ResidualConvUnit(channels)
    x = torch.randn(batch_size, channels, height, width)

    out = rcu(x)

    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"
    assert not torch.isnan(out).any(), "Output contains NaNs"
    print(f"ResidualConvUnit: {x.shape} -> {out.shape} ✓")


def test_fusion_block():
    """Test FusionBlock upsamples by 2x."""
    batch_size = 2
    channels = 256
    height, width = 48, 48

    fusion = FusionBlock(channels)
    x = torch.randn(batch_size, channels, height, width)

    out = fusion(x)

    expected_shape = (batch_size, channels, height * 2, width * 2)
    assert out.shape == expected_shape, f"Expected {expected_shape}, got {out.shape}"
    assert not torch.isnan(out).any(), "Output contains NaNs"
    print(f"FusionBlock: {x.shape} -> {out.shape} ✓")


def test_dpt_pointmap_head_shapes():
    """Test PointMapHead produces correct output shapes."""
    batch_size = 2
    num_views = 3
    img_size = 384
    patch_size = 8
    embed_dim = 1024

    patch_grid = img_size // patch_size  # 48
    num_patches = patch_grid * patch_grid  # 2304

    # Simulate decoder output: (B*V, P, C)
    BV = batch_size * num_views
    tokens = torch.randn(BV, num_patches, embed_dim)

    head = PointMapHead(
        in_dim=embed_dim,
        hidden_dim=256,
        img_size=img_size,
        patch_size=patch_size
    )

    pointmap, confidence = head(tokens)

    # Expected shapes
    expected_pointmap_shape = (BV, img_size * img_size, 3)  # (6, 147456, 3)
    expected_confidence_shape = (BV, img_size * img_size, 1)  # (6, 147456, 1)

    assert pointmap.shape == expected_pointmap_shape, \
        f"Pointmap: expected {expected_pointmap_shape}, got {pointmap.shape}"
    assert confidence.shape == expected_confidence_shape, \
        f"Confidence: expected {expected_confidence_shape}, got {confidence.shape}"
    assert not torch.isnan(pointmap).any(), "Pointmap contains NaNs"
    assert not torch.isnan(confidence).any(), "Confidence contains NaNs"

    print(f"PointMapHead input: {tokens.shape}")
    print(f"PointMapHead pointmap output: {pointmap.shape} ✓")
    print(f"PointMapHead confidence output: {confidence.shape} ✓")


def test_dpt_pointmap_head_small():
    """Test PointMapHead with smaller image size (for faster testing)."""
    batch_size = 2
    num_views = 2
    img_size = 128
    patch_size = 8
    embed_dim = 128

    patch_grid = img_size // patch_size  # 16
    num_patches = patch_grid * patch_grid  # 256

    BV = batch_size * num_views
    tokens = torch.randn(BV, num_patches, embed_dim)

    head = PointMapHead(
        in_dim=embed_dim,
        hidden_dim=64,
        img_size=img_size,
        patch_size=patch_size
    )

    pointmap, confidence = head(tokens)

    expected_pointmap_shape = (BV, img_size * img_size, 3)  # (4, 16384, 3)
    expected_confidence_shape = (BV, img_size * img_size, 1)  # (4, 16384, 1)

    assert pointmap.shape == expected_pointmap_shape, \
        f"Pointmap: expected {expected_pointmap_shape}, got {pointmap.shape}"
    assert confidence.shape == expected_confidence_shape, \
        f"Confidence: expected {expected_confidence_shape}, got {confidence.shape}"

    print(f"PointMapHead (small) input: {tokens.shape}")
    print(f"PointMapHead (small) pointmap output: {pointmap.shape} ✓")
    print(f"PointMapHead (small) confidence output: {confidence.shape} ✓")


def test_dpt_pointmap_head_gradient():
    """Test that gradients flow through PointMapHead."""
    batch_size = 1
    img_size = 64
    patch_size = 8
    embed_dim = 64

    patch_grid = img_size // patch_size  # 8
    num_patches = patch_grid * patch_grid  # 64

    tokens = torch.randn(batch_size, num_patches, embed_dim, requires_grad=True)

    head = PointMapHead(
        in_dim=embed_dim,
        hidden_dim=32,
        img_size=img_size,
        patch_size=patch_size
    )

    pointmap, confidence = head(tokens)

    # Compute dummy loss and backprop
    loss = pointmap.sum() + confidence.sum()
    loss.backward()

    assert tokens.grad is not None, "Gradients should flow to input tokens"
    assert not torch.isnan(tokens.grad).any(), "Gradients contain NaNs"
    print("Gradient flow test passed ✓")


if __name__ == "__main__":
    print("=" * 50)
    print("Testing ResidualConvUnit")
    print("=" * 50)
    test_residual_conv_unit()

    print("\n" + "=" * 50)
    print("Testing FusionBlock")
    print("=" * 50)
    test_fusion_block()

    print("\n" + "=" * 50)
    print("Testing PointMapHead (small)")
    print("=" * 50)
    test_dpt_pointmap_head_small()

    print("\n" + "=" * 50)
    print("Testing PointMapHead gradient flow")
    print("=" * 50)
    test_dpt_pointmap_head_gradient()

    print("\n" + "=" * 50)
    print("Testing PointMapHead (full size)")
    print("=" * 50)
    test_dpt_pointmap_head_shapes()

    print("\n" + "=" * 50)
    print("All tests passed!")
    print("=" * 50)
