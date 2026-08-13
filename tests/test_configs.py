"""The shipped configs must agree with what the model can actually build.

configs/train_mini.yaml omitted `patch_size` and picked up scripts/train.py's default
of 8, which cannot load the ViT-L/16 DUSt3R encoder at all - `make train-debug` died at
model construction. Nothing caught it because no test read the configs.
"""

import sys
from pathlib import Path

import pytest
import yaml

root_path = Path(__file__).parent.parent
sys.path.append(str(root_path))

from models.encoder_vit import DUST3R_ENCODER

CONFIG_DIR = root_path / "configs"
CONFIGS = sorted(CONFIG_DIR.glob("*.yaml"))
EVAL_CONFIGS = sorted(CONFIG_DIR.glob("evaluate*.yaml"))


def load(path):
    with open(path) as f:
        return yaml.safe_load(f)


def test_configs_exist():
    assert CONFIGS, f"no configs found in {CONFIG_DIR}"
    assert EVAL_CONFIGS, f"no evaluate*.yaml found in {CONFIG_DIR}"


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_patch_size_never_contradicts_the_encoder(path):
    """patch_size is pinned by the DUSt3R encoder. A config may omit it - the scripts
    take it from DUST3R_ENCODER - but must never set it to something else."""
    patch_size = load(path).get("patch_size")
    if patch_size is not None:
        assert patch_size == DUST3R_ENCODER["patch_size"], (
            f"{path.name} sets patch_size={patch_size}, encoder is fixed at "
            f"{DUST3R_ENCODER['patch_size']}"
        )


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_image_size_is_divisible_by_the_patch_size(path):
    """A non-divisible image_size gives a fractional patch grid, so the raymap and
    pointmap heads silently read off a grid that does not tile the image."""
    image_size = load(path).get("image_size")
    if image_size is None:
        return

    patch = DUST3R_ENCODER["patch_size"]
    assert len(image_size) == 2, f"{path.name}: image_size must be [H, W], got {image_size}"
    for side in image_size:
        assert side % patch == 0, f"{path.name}: image_size {image_size} not divisible by {patch}"


@pytest.mark.parametrize("path", EVAL_CONFIGS, ids=lambda p: p.name)
def test_eval_configs_carry_their_geometry(path):
    """Evaluation geometry used to be hardcoded in the scripts while only `checkpoint`
    and `data` came from config, so evaluating a model trained at another resolution
    meant editing source."""
    cfg = load(path)
    for key in ("checkpoint", "data", "n_frames", "image_size"):
        assert key in cfg, f"{path.name} is missing '{key}'"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
