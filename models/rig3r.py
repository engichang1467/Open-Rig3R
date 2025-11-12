import torch
import torch.nn as nn

from models.encoder_vit import ViTEncoder
# from models.heads.pointmap_head import PointMapHead 
# from models.heads.rig_raymap_head import RigRaymapHead
# from models.heads.pose_raymap_head import PoseRaymapHead
from models.decoder_transformer import RigAwareTransformerDecoder


class Rig3R(nn.Module):
    """
        Main Rig3R model: Encoder → Rig-Aware Transformer Decoder → Heads
    """
    def __init__(
        self,
        encoder_ckpt=None,
        img_size=384,
        patch_size=8,
        embed_dim=1024,
        metadata_dim=64,
        num_decoder_layers=6,
        num_heads=8,
        mlp_dim=2048
    ):
        super().__init__()

        # --- Encoder ---
        self.encoder = ViTEncoder(
            checkpoint_path=encoder_ckpt,
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim
        )

        # --- Rig-aware Transformer Decoder ---
        self.decoder = RigAwareTransformerDecoder(
            embed_dim=embed_dim,
            num_layers=num_decoder_layers,
            num_heads=num_heads,
            metadata_dim=metadata_dim,
            mlp_dim=mlp_dim
        )

        # # --- Heads ---
        # self.pointmap_head = PointMapHead(in_dim=embed_dim)
        # self.pose_head = PoseRaymapHead(in_dim=embed_dim)
        # self.rig_head = RigRaymapHead(in_dim=embed_dim)

    
    def forward(self, images, metadata=None):
        """
            Args:
                images: tensor(B, N, 3, H, W) - batch of N images per sample
                metadata: dict containing optional rig or camera info
            Returns:
                dict of predictions: pointmap, pose_raymap, rig_raymap
        """

        B, N, C, H, W = images.shape

        # --- Encode each image ---
        tokens_list = []
        for i in range(N):
            enc_out = self.encoder(images[:, i])
            tokens_list.append(enc_out["tokens"]) # (B, num_patches, C)

        # --- Concatenate tokens from all views ---
        joint_tokens = torch.cat(tokens_list, dim=1) # (B, N * num_patches, C)

        # --- Decode with rig-aware transformer ---
        dec_tokens = self.decoder(joint_tokens, frames=N, metadata=metadata, cam2rig=metadata["cam2rig"] if metadata else None) # (B, N * num_patches, C)

        # # --- Apply heads --- 
        # pointmap = self.pointmap_head(dec_tokens["pointmap"])
        # pose_raymap = self.pose_head(dec_tokens["pose_raymap"])
        # rig_raymap = self.rig_head(dec_tokens["rig_raymap"], cam2rig=metadata.get("cam2rig") if metadata else None)

        # return {
        #     "pointmap": pointmap,
        #     "pose_raymap": pose_raymap,
        #     "rig_raymap": rig_raymap
        # }

        return dec_tokens