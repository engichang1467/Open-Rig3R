import torch
import argparse
import timm
import torch.nn as nn

# allow argparse.Namespace inside the checkpoint
torch.serialization.add_safe_globals([argparse.Namespace])

# DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth: enc_depth=24, enc_embed_dim=1024,
# enc_num_heads=16, patch_size=16. Anything else will fail the strict load below.
DUST3R_ENCODER = dict(patch_size=16, embed_dim=1024, depth=24, num_heads=16)


class ViTEncoder(nn.Module):
    """ViT-L/16 image encoder, optionally initialised from DUSt3R's CroCo encoder.

    DUSt3R's encoder uses timm's block naming under an ``enc_`` prefix, so the
    checkpoint drops straight into a timm ``VisionTransformer`` after a prefix
    rename - no per-tensor remap, no qkv surgery.
    """

    def __init__(
        self,
        checkpoint_path=None,
        img_size=384,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        freeze=True,
    ):
        super().__init__()
        self.vit = timm.models.VisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            class_token=False,
            global_pool="",
            num_classes=0,
            # ponytail: no positional signal at all right now, so the encoder is
            # permutation-equivariant over patches. Rig3R sec 3.3 specifies 2D
            # sine-cosine (not the checkpoint's RoPE100) - see issue #42.
            pos_embed="none",
        )

        if checkpoint_path is not None:
            self._load_dust3r_weights(checkpoint_path)

        # 300M frozen params cost ~1.2 GiB of activations for 128 views; trainable
        # they cost ~5 GiB of optimizer state on top and will not fit a 12 GB card.
        if freeze:
            self.vit.requires_grad_(False)

    def _load_dust3r_weights(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        # DUSt3R wraps the weights: the file is {"model": ..., "args": ...}
        state_dict = checkpoint.get("model", checkpoint)

        # CroCo: patch_embed.* / enc_blocks.N.* / enc_norm.*
        # timm:  patch_embed.* / blocks.N.*     / norm.*
        encoder_weights = {
            key.replace("enc_blocks.", "blocks.").replace("enc_norm.", "norm."): value
            for key, value in state_dict.items()
            if key.startswith(("patch_embed.", "enc_blocks.", "enc_norm."))
        }
        if not encoder_weights:
            raise ValueError(
                f"No encoder weights (patch_embed./enc_blocks./enc_norm.) in {checkpoint_path}. "
                f"Top-level keys: {sorted(state_dict)[:10]}"
            )

        # strict: a silent no-op load is the bug this replaces. Any name or shape
        # drift - wrong embed_dim, wrong patch_size - must raise, not print.
        self.vit.load_state_dict(encoder_weights, strict=True)
        print(f"Loaded {len(encoder_weights)} DUSt3R encoder tensors from {checkpoint_path}")

    def forward(self, x):
        # forward_features applies patch embed + blocks + the loaded enc_norm
        return {"tokens": self.vit.forward_features(x)}  # (B, P, C)
