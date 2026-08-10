import torch
import sys
from pathlib import Path

root_path = Path(__file__).parent.parent
sys.path.append(str(root_path))

from models.decoder_transformer import RigAwareTransformerDecoder

# quick smoke test
torch.manual_seed(0)
B = 2
V = 2            # frames/views
P = 16           # patches per frame (small for test) = 4x4 grid
C = 64           # embed dim for quick test
img_size = 32    # 4x4 patches with patch_size=8
patch_size = 8
tokens = torch.randn(B, V * P, C)

# instantiate decoder with small dims for test
decoder = RigAwareTransformerDecoder(
    embed_dim=C,
    num_layers=4,
    num_heads=4,
    mlp_dim=C * 4,
    attn_dropout=0.0,
    img_size=img_size,
    patch_size=patch_size
)

# dummy metadata: the per-view fields of sec 3.3
metadata = {
    "frame_index": torch.arange(V).expand(B, V),
    "camera_id": torch.arange(V).expand(B, V) % 2,
    "timestamp": torch.zeros(B, V),
}

out = decoder(tokens, frames=V, metadata=metadata)
print("pointmap shape:", out["pointmap"].shape)    # (B, V, H*W, 3) dense predictions
print("pointmap_conf shape:", out["pointmap_conf"].shape)  # (B, V, H*W, 1) confidence
print("pose_raymap shape:", out["pose_raymap"].shape)  # (B, V, P, 3)
print("rig_raymap shape:", out["rig_raymap"].shape)    # (B, V, P, 6)
print("features shape:", out["features"].shape)    # (B, V, P, C)