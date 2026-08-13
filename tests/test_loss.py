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


def test_half_precision_predictions_backward():
    """Under autocast the model emits half, targets stay fp32.

    F.mse_loss refuses that pair, so the camera-centre branches blew up the moment
    they stopped being dead code.
    """
    torch.manual_seed(0)
    preds, gts = sample_batch()
    preds = {k: v.half().requires_grad_() for k, v in preds.items()}

    total, _ = MultiTaskLoss()(preds, gts)
    total.backward()

    assert total.dtype == torch.float32, f"loss should reduce in fp32, got {total.dtype}"
    for key, value in preds.items():
        assert value.grad is not None, f"{key} got no gradient"
        assert value.grad.dtype == torch.float16, f"{key} grad is {value.grad.dtype}"

    print(f"half predictions -> fp32 loss {total.item():.4f}, half grads test passed!")


def test_direction_term_is_the_ray_norm_not_cosine():
    """Eq. 4 uses ||r - r_bar||. For unit vectors that is sqrt(2 - 2cos).

    The distinction matters near the optimum: 1-cos falls off quadratically and its
    gradient fades, the norm stays linear.
    """
    torch.manual_seed(0)
    directions = torch.nn.functional.normalize(torch.randn(1, 1, 4, 3), dim=-1)
    perturbed = torch.nn.functional.normalize(directions + 0.3 * torch.randn(1, 1, 4, 3), dim=-1)
    center = torch.zeros(1, 1, 3)

    preds = {'rig_raymap': raymap(center, perturbed), 'camera_center_rig': center}
    gts = {'rig_raymap': raymap(center, directions), 'camera_center_rig': center}

    _, loss_dict = MultiTaskLoss(beta=0.0)(preds, gts)

    cosine = torch.nn.functional.cosine_similarity(perturbed, directions, dim=-1)
    expected = (2 - 2 * cosine).clamp(min=0).sqrt().mean()
    torch.testing.assert_close(loss_dict['rig_raymap'], expected)

    cosine_loss = (1 - cosine).mean()
    print(f"ray norm {expected:.4f} vs cosine distance {cosine_loss:.4f} test passed!")


def test_matching_directions_score_zero():
    center = torch.zeros(1, 1, 3)
    directions = torch.nn.functional.normalize(torch.randn(1, 1, 4, 3), dim=-1)
    same = {'rig_raymap': raymap(center, directions), 'camera_center_rig': center}

    _, loss_dict = MultiTaskLoss()(dict(same), same)
    torch.testing.assert_close(loss_dict['rig_raymap'], torch.tensor(0.0))
    print("A perfect raymap scores zero test passed!")


def test_confidence_regularizer_punishes_giving_up():
    """-alpha*log(C) is what stops the model zeroing its confidence to escape Eq. 3."""
    torch.manual_seed(0)
    point_pred = torch.randn(1, 1, 8, 3)
    gts = {'pointmap': point_pred + 0.5, 'pointmap_conf': torch.ones(1, 1, 8)}

    criterion = MultiTaskLoss(alpha=0.2)
    losses = {}
    for name, value in (("giving up", 1.0), ("confident", 8.0)):
        preds = {'pointmap': point_pred, 'pointmap_conf': torch.full((1, 1, 8, 1), value)}
        _, loss_dict = criterion(preds, gts)
        losses[name] = loss_dict['pointmap']
        print(f"C={value}: pointmap loss {losses[name]:.4f}")

    # with a real error present, high confidence should cost more, not less
    assert losses["confident"] > losses["giving up"], "confidence is not weighting the error"

    # and without the regularizer nothing stops C collapsing
    no_reg = MultiTaskLoss(alpha=0.0)
    tiny = {'pointmap': point_pred, 'pointmap_conf': torch.full((1, 1, 8, 1), 1e-4)}
    _, unregularized = no_reg(tiny, gts)
    _, regularized = criterion(tiny, gts)
    assert regularized['pointmap'] > unregularized['pointmap'], (
        "alpha must penalise vanishing confidence"
    )
    print("Confidence regularizer punishes giving up test passed!")


def test_confidence_comes_from_the_prediction():
    """Eq. 3's C is the model's own output, not the lidar validity mask."""
    torch.manual_seed(0)
    point_pred = torch.randn(1, 1, 8, 3)
    gts = {'pointmap': point_pred + 0.5, 'pointmap_conf': torch.ones(1, 1, 8)}
    criterion = MultiTaskLoss()

    base = {'pointmap': point_pred, 'pointmap_conf': torch.full((1, 1, 8, 1), 2.0)}
    moved = {'pointmap': point_pred, 'pointmap_conf': torch.full((1, 1, 8, 1), 4.0)}

    _, a = criterion(base, gts)
    _, b = criterion(moved, gts)
    assert not torch.isclose(a['pointmap'], b['pointmap']), (
        "predicted confidence does not affect the loss - still reading the GT mask"
    )
    print("Predicted confidence drives the pointmap term test passed!")


def test_invalid_patches_supervise_nothing():
    """D_v in Eq. 3 is the valid set; patches with no lidar return are excluded."""
    torch.manual_seed(0)
    point_pred = torch.randn(1, 1, 4, 3)
    conf = torch.ones(1, 1, 4, 1)

    gt = point_pred.clone()
    gt[0, 0, 3] += 100.0  # a wild error, but in a patch with no coverage

    mask = torch.tensor([[[1.0, 1.0, 1.0, 0.0]]])
    preds = {'pointmap': point_pred, 'pointmap_conf': conf}
    _, loss_dict = MultiTaskLoss(alpha=0.0)({**preds}, {'pointmap': gt, 'pointmap_conf': mask})

    torch.testing.assert_close(loss_dict['pointmap'], torch.tensor(0.0))
    print("Masked-out patches contribute nothing test passed!")


def test_raw_error_is_reported_apart_from_the_confidence():
    """The bundled term can fall while reconstruction is unchanged; err must not."""
    torch.manual_seed(0)
    point_pred = torch.randn(1, 1, 8, 3)
    # A near-zero error is what lets -alpha*log(C) outrun C*error; that is the regime
    # waymo_mini reaches by epoch 7.
    gts = {'pointmap': point_pred + 1e-4, 'pointmap_conf': torch.ones(1, 1, 8)}
    criterion = MultiTaskLoss(alpha=0.2)

    reported = {}
    for value in (2.0, 1000.0):
        preds = {'pointmap': point_pred, 'pointmap_conf': torch.full((1, 1, 8, 1), value)}
        reported[value] = criterion(preds, gts)[1]

    # Inflating C alone drives the bundled term down, and below zero. That is the whole
    # bug: it must not look like reconstruction improved.
    assert reported[1000.0]['pointmap'] < reported[2.0]['pointmap'] < 0
    torch.testing.assert_close(
        reported[1000.0]['pointmap_err'], reported[2.0]['pointmap_err']
    )
    assert reported[1000.0]['pointmap_err'] > 0
    torch.testing.assert_close(
        reported[1000.0]['pointmap_conf_mean'], torch.tensor(1000.0)
    )
    print("Raw error and confidence are reported separately test passed!")


if __name__ == "__main__":
    test_loss_smoke()
    test_raw_error_is_reported_apart_from_the_confidence()
    test_half_precision_predictions_backward()
    test_direction_term_is_the_ray_norm_not_cosine()
    test_matching_directions_score_zero()
    test_confidence_regularizer_punishes_giving_up()
    test_confidence_comes_from_the_prediction()
    test_invalid_patches_supervise_nothing()
    test_camera_center_terms_contribute()
    test_direction_term_ignores_the_centre()
