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
P = 16           # patches per frame (small for test)
C = 64           # embed dim for quick test
tokens = torch.randn(B, V * P, C)

# instantiate decoder with small dims for test
decoder = RigAwareTransformerDecoder(
    embed_dim=C,
    num_layers=4,
    num_heads=4,
    mlp_dim=C * 4,
    metadata_dim=None,
    metadata_tokens=2,
    metadata_dropout=0.3,
    attn_dropout=0.0
)

# dummy metadata: (B, M, metadata_dim)
# meta = torch.randn(B, 2, 32)
# dummy metadata: dictionary with cam2rig
metadata = {
    "cam2rig": torch.randn(B, 3, 3)  # random rotation matrices for test
}

out = decoder(tokens, frames=V, metadata=metadata, cam2rig=metadata["cam2rig"])
print("pointmap shape:", out["pointmap"].shape)    # (B, V, P, 3)
print("pose_raymap shape:", out["pose_raymap"].shape)
print("rig_raymap shape:", out["rig_raymap"].shape)
print("features shape:", out["features"].shape)    # (B, V, P, C)