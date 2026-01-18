import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class ResidualConvUnit(nn.Module):
    """
    Residual convolutional unit for feature refinement.

    Structure:
        Input → Conv3x3 → ReLU → Conv3x3 → Add → Output
          └──────────────────────────────────┘ (skip)
    """
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=False)  # inplace=False required for gradient checkpointing

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out = out + residual
        return self.relu(out)


class FusionBlock(nn.Module):
    """
    Fusion block that refines features and upsamples by 2x.

    Structure:
        Input → ResidualConvUnit → ResidualConvUnit → Upsample 2x → Output
    """
    def __init__(self, channels):
        super().__init__()
        self.rcu1 = ResidualConvUnit(channels)
        self.rcu2 = ResidualConvUnit(channels)

    def forward(self, x):
        out = self.rcu1(x)
        out = self.rcu2(out)
        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=True)
        return out


class PointMapHead(nn.Module):
    """
    DPT-style head for dense pointmap prediction.

    Takes patch tokens, reshapes to spatial grid, and progressively
    upsamples to full image resolution using fusion blocks.

    Input:  tokens (B*V, P, C) where P = (img_size/patch_size)^2
    Output: pointmap (B*V, H, W, 3), confidence (B*V, H, W, 1)
    """
    def __init__(self, in_dim=1024, hidden_dim=256, img_size=384, patch_size=8, use_gradient_checkpointing=True):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.img_size = img_size
        self.patch_size = patch_size
        self.patch_grid = img_size // patch_size  # e.g., 384/8 = 48
        self.use_gradient_checkpointing = use_gradient_checkpointing

        # Project from token dim to hidden dim
        self.input_proj = nn.Conv2d(in_dim, hidden_dim, kernel_size=1)

        # 3 Fusion blocks for 8x upsampling (48 → 96 → 192 → 384)
        self.fusion1 = FusionBlock(hidden_dim)
        self.fusion2 = FusionBlock(hidden_dim)
        self.fusion3 = FusionBlock(hidden_dim)

        # Output heads: 3 channels for XYZ pointmap, 1 channel for confidence
        self.output_conv = nn.Conv2d(hidden_dim, 4, kernel_size=1)

    def forward(self, tokens):
        """
        Args:
            tokens: (B*V, P, C) where P = patch_grid^2, C = in_dim

        Returns:
            pointmap: (B*V, H*W, 3) dense 3D points
            confidence: (B*V, H*W, 1) per-pixel confidence
        """
        BV, P, C = tokens.shape
        H_p = W_p = self.patch_grid  # patch grid size (e.g., 48)
        H = W = self.img_size  # output image size (e.g., 384)

        # Reshape tokens to spatial grid: (B*V, P, C) → (B*V, C, H_p, W_p)
        x = tokens.permute(0, 2, 1)  # (B*V, C, P)
        x = x.view(BV, C, H_p, W_p)  # (B*V, C, 48, 48)

        # Project to hidden dimension
        x = self.input_proj(x)  # (B*V, hidden_dim, 48, 48)

        # Progressive upsampling through fusion blocks
        # Use gradient checkpointing during training to reduce memory usage
        if self.training and self.use_gradient_checkpointing:
            x = checkpoint(self.fusion1, x, use_reentrant=False)  # (B*V, hidden_dim, 96, 96)
            x = checkpoint(self.fusion2, x, use_reentrant=False)  # (B*V, hidden_dim, 192, 192)
            x = checkpoint(self.fusion3, x, use_reentrant=False)  # (B*V, hidden_dim, 384, 384)
        else:
            x = self.fusion1(x)  # (B*V, hidden_dim, 96, 96)
            x = self.fusion2(x)  # (B*V, hidden_dim, 192, 192)
            x = self.fusion3(x)  # (B*V, hidden_dim, 384, 384)

        # Output projection
        out = self.output_conv(x)  # (B*V, 4, 384, 384)

        # Split into pointmap (3 channels) and confidence (1 channel)
        pointmap = out[:, :3, :, :]  # (B*V, 3, H, W)
        confidence = out[:, 3:4, :, :]  # (B*V, 1, H, W)

        # Reshape to (B*V, H*W, channels) format to match expected output
        pointmap = pointmap.permute(0, 2, 3, 1)  # (B*V, H, W, 3)
        pointmap = pointmap.reshape(BV, H * W, 3)  # (B*V, H*W, 3)

        confidence = confidence.permute(0, 2, 3, 1)  # (B*V, H, W, 1)
        confidence = confidence.reshape(BV, H * W, 1)  # (B*V, H*W, 1)

        return pointmap, confidence
