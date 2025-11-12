import torch
import argparse
from pathlib import Path

import sys

root_path = Path(__file__).parent.parent
sys.path.append(str(root_path))

# allow argparse.Namespace inside the checkpoint
torch.serialization.add_safe_globals([argparse.Namespace])

ckpt_path = Path.cwd().joinpath("checkpoints/pretrained/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth")
ckpt = torch.load(ckpt_path, map_location="cpu")
# ckpt = torch.load("checkpoints/pretrained/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth", map_location="cpu")

# print(ckpt.keys())
# print(list(ckpt.keys()))
print(list(ckpt.keys())[:10])
print(list(ckpt["model"].keys())[:15])

print('pos_embed' in ckpt['model'])

print([k for k in ckpt['model'].keys() if 'pos' in k])