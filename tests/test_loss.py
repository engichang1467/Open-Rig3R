import torch
import sys
from pathlib import Path

root_path = Path(__file__).parent.parent
sys.path.append(str(root_path))

from models.losses import MultiTaskLoss

B, V, P = 2, 3, 16  # batch, views, patches


def raymap(center, directions):
    """(B, V, 3) centre + (B, V, P, 3) directions -> (B, V, P, 6), as the head emits."""
    directions = torch.nn.functional.normalize(directions, dim=-1)
    return torch.cat([center.unsqueeze(2).expand_as(directions), directions], dim=-1)


def sample_batch():
    preds, gts = {}, {}
    preds['pointmap'] = torch.randn(B, V, P, 3)
    gts['pointmap'] = torch.randn(B, V, P, 3)
    gts['pointmap_conf'] = torch.rand(B, V, P)

    for name in ('pose', 'rig'):
        key = f'{name}_raymap'
        center_key = f'camera_center_{name}'
        preds[center_key] = torch.randn(B, V, 3)
        gts[center_key] = torch.randn(B, V, 3)
        preds[key] = raymap(preds[center_key], torch.randn(B, V, P, 3))
        gts[key] = raymap(gts[center_key], torch.randn(B, V, P, 3))

    return preds, gts


def test_loss_smoke():
    torch.manual_seed(0)
    preds, gts = sample_batch()

    total, loss_dict = MultiTaskLoss()(preds, gts)

    print("Total loss:", total.item())
    for k, v in loss_dict.items():
        print(f"{k}: {v}")

    assert not torch.isnan(total), "Loss contains NaNs"
    print("Loss smoke test passed!")


def test_camera_center_terms_contribute():
    """The centre branches were unreachable before #40 - they must fire now."""
    torch.manual_seed(0)
    preds, gts = sample_batch()

    criterion = MultiTaskLoss()
    _, with_center = criterion(preds, gts)

    # same batch, but the centre prediction is exactly right
    for name in ('pose', 'rig'):
        preds[f'camera_center_{name}'] = gts[f'camera_center_{name}'].clone()
    _, exact_center = criterion(preds, gts)

    for name in ('pose', 'rig'):
        key = f'{name}_raymap'
        drop = with_center[key] - exact_center[key]
        print(f"{key}: {with_center[key]:.4f} -> {exact_center[key]:.4f} (centre term {drop:.4f})")
        assert drop > 1e-3, f"{key} ignores the camera centre - the branch is still dead"

    print("Camera centre terms contribute test passed!")


def test_direction_term_ignores_the_centre():
    """Channels 0:3 are scored by the centre term only, never twice."""
    torch.manual_seed(0)
    preds, gts = sample_batch()
    criterion = MultiTaskLoss()
    _, before = criterion(preds, gts)

    # move the centre channels of the raymap without touching the centre prediction
    for name in ('pose', 'rig'):
        preds[f'{name}_raymap'][..., :3] += 10.0
    _, after = criterion(preds, gts)

    for name in ('pose', 'rig'):
        key = f'{name}_raymap'
        torch.testing.assert_close(before[key], after[key])

    print("Direction term ignores the centre channels test passed!")


if __name__ == "__main__":
    test_loss_smoke()
    test_camera_center_terms_contribute()
    test_direction_term_ignores_the_centre()
