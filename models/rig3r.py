import torch.nn as nn

from models.encoder_vit import ViTEncoder
from models.decoder_transformer import RigAwareTransformerDecoder


class Rig3R(nn.Module):
    """
        Main Rig3R model: Encoder → Rig-Aware Transformer Decoder → Heads
    """
    def __init__(
        self,
        encoder_ckpt=None,
        img_size=384,
        patch_size=16,
        embed_dim=1024,
        num_decoder_layers=6,
        num_heads=8,
        mlp_dim=2048,
        metadata_dropout=0.5,
        freeze_encoder=True
    ):
        super().__init__()

        # --- Encoder ---
        self.encoder = ViTEncoder(
            checkpoint_path=encoder_ckpt,
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            freeze=freeze_encoder
        )

        # --- Rig-aware Transformer Decoder ---
        self.decoder = RigAwareTransformerDecoder(
            embed_dim=embed_dim,
            num_layers=num_decoder_layers,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            metadata_dropout=metadata_dropout,
            img_size=img_size,
            patch_size=patch_size
        )
    
    def forward(self, images, metadata=None):
        """
            Args:
                images: tensor(B, N, 3, H, W) - batch of N images per sample
                metadata: dict containing optional rig or camera info
            Returns:
                dict of predictions: pointmap, pose_raymap, rig_raymap
        """

        B, N, C, H, W = images.shape

        # --- Encode every view in one pass ---
        tokens = self.encoder(images.reshape(B * N, C, H, W))["tokens"]   # (B*N, num_patches, C)

        # --- Concatenate tokens from all views ---
        joint_tokens = tokens.reshape(B, N * tokens.shape[1], tokens.shape[2])

        # --- Decode with rig-aware transformer ---
        dec_tokens = self.decoder(joint_tokens, frames=N, metadata=metadata) # (B, N * num_patches, C)

        return dec_tokens