import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiTaskLoss(nn.Module):
    """
        Rig3R Eq. 5: L_total = L_pmap + lambda_p * L_p_rmap + lambda_r * L_r_rmap

        Eq. 3  L_pmap = sum_{i in D_v} C_i * ||X_i - X_bar_i / z_bar|| - alpha log C_i
        Eq. 4  L_rmap = sum_hw ||r - r_bar|| + beta * ||c - c_bar / z_bar||

        Both equations use an unsquared Euclidean norm, and both expect ground truth
        already divided by z_bar - see utils.raymap.scene_scale. C is the model's own
        predicted confidence, not a validity mask.
    """
    def __init__(self, w_point=1.0, w_pose=1.0, w_rig=1.0, alpha=0.2, beta=1.0,
                 reduction='mean'):
        super().__init__()
        self.w_point = w_point
        self.w_pose = w_pose
        self.w_rig = w_rig
        self.alpha = alpha  # Eq. 3 confidence regularizer; paper gives no value
        self.beta = beta    # Eq. 4 camera centre weight; paper gives no value
        self.reduction = reduction

    def forward(self, preds, gts):
        """
            Args:
                preds: dict with keys ['pointmap', 'pose_raymap', 'rig_raymap']
                gts:   dict with same keys
            Returns:
                total_loss, loss_dict
        """

        loss_dict = {}

        # Predictions arrive in whatever dtype autocast produced, targets are always
        # fp32. F.mse_loss rejects a half/float pair outright ("Found dtype Float but
        # expected Half"), and loss reductions are safer accumulated at full precision
        # anyway, so score everything in fp32. Casting back is differentiable.
        preds = {k: v.float() if torch.is_tensor(v) else v for k, v in preds.items()}
        gts = {k: v.float() if torch.is_tensor(v) else v for k, v in gts.items()}

        # ==========================================================
        # 1. Confidence‑weighted Pointmap loss (L2)
        # gts must include: 'pointmap' and 'pointmap_conf'
        # ==========================================================
        if 'pointmap' in preds and 'pointmap' in gts:
            # Eq. 3 is a norm, not a squared error
            error = (preds['pointmap'] - gts['pointmap']).norm(dim=-1)  # (B, V, P)

            conf = preds.get('pointmap_conf')
            if conf is not None:
                # C is the model's own confidence, so it can downweight what it cannot
                # see. -alpha*log(C) is what stops it driving C to zero to escape the
                # loss entirely.
                conf = conf.squeeze(-1) if conf.dim() == error.dim() + 1 else conf
                term = conf * error - self.alpha * conf.log()
            else:
                term = error

            # gts['pointmap_conf'] is the lidar validity fraction: the set D_v Eq. 3
            # sums over, not a weight. Patches with no return supervise nothing.
            loss_point = self._reduce_masked(term, gts.get('pointmap_conf'))
            loss_dict['pointmap'] = loss_point
        else:
            loss_point = 0.0


        # ==========================================================
        # 2. Pose raymap loss = direction loss + camera center loss
        # ==========================================================
        if 'pose_raymap' in preds and 'pose_raymap' in gts:
            loss_pose = self._raymap_loss(
                preds, gts, 'pose_raymap', 'camera_center_pose', 'pose', loss_dict
            )
            loss_dict['pose_raymap'] = loss_pose
        else:
            loss_pose = 0.0


        # ==========================================================
        # 3. Rig raymap loss = direction loss + camera center loss
        # ==========================================================
        if 'rig_raymap' in preds and 'rig_raymap' in gts:
            loss_rig = self._raymap_loss(
                preds, gts, 'rig_raymap', 'camera_center_rig', 'rig', loss_dict
            )
            loss_dict['rig_raymap'] = loss_rig
        else:
            loss_rig = 0.0

        # --- Combine ---
        total = (
            self.w_point * loss_point +
            self.w_pose  * loss_pose +
            self.w_rig   * loss_rig
        )
        loss_dict['total'] = total
        return total, loss_dict
    
    def _raymap_loss(self, preds, gts, raymap_key, center_key, name, loss_dict):
        """Eq. 4: ||r - r_bar|| over patches, plus beta * ||c - c_bar / z_bar||.

        Channels 0:3 of the raymap are the shared camera centre, scored by the centre
        term; including them in the direction term would count them twice. The norm
        rather than a cosine is deliberate - for unit vectors ||r - r_bar|| is
        sqrt(2 - 2cos), whose gradient stays linear near the optimum where cosine's
        goes quadratic and fades out.

        The two halves are reported separately in loss_dict: bundled into one number
        they cannot be compared across a change to either term, because the direction
        and centre parts move on completely different scales.
        """
        direction = (preds[raymap_key][..., 3:] - gts[raymap_key][..., 3:]).norm(dim=-1)
        loss = self._reduce(direction)
        loss_dict[f'{name}_dir'] = loss

        if center_key in preds and center_key in gts:
            center = self._reduce((preds[center_key] - gts[center_key]).norm(dim=-1))
            loss_dict[f'{name}_center'] = center
            loss = loss + self.beta * center

        return loss

    def _reduce_masked(self, loss, mask):
        """Reduce over the valid entries only, so empty patches supervise nothing."""
        if mask is None:
            return self._reduce(loss)

        mask = mask.squeeze(-1) if mask.dim() == loss.dim() + 1 else mask
        valid = mask > 0
        if not valid.any():
            return loss.sum() * 0.0  # keeps the graph connected with nothing to learn

        return loss[valid].sum() if self.reduction == 'sum' else loss[valid].mean()

    def _reduce(self, loss):
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss
