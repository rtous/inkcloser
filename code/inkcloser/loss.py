"""Training loss (CODE BLOCK 10 of the notebook)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LineLoss(nn.Module):
    """Hole-weighted BCE + global Dice + gap Dice (paper Sec. 5.2)."""

    def __init__(self, hole_weight=3.0, bce_weight=1.0, dice_weight=1.0,
                 gap_dice_weight=0.5, gap_radius=7, smooth=1.0):
        super().__init__()
        self.hole_weight, self.bce_weight = hole_weight, bce_weight
        self.dice_weight, self.gap_dice_weight = dice_weight, gap_dice_weight
        self.gap_radius, self.smooth = gap_radius, smooth
        self.last_components = {}

    @staticmethod
    def _hole_mask(corrupted, target):
        return ((corrupted > 0.9) & (target < 0.5)).float()

    def _dice(self, p, t, region=None):
        if region is not None:
            p, t = p * region, t * region
        dims = (1, 2, 3)
        inter = (p * t).sum(dims)
        card = p.sum(dims) + t.sum(dims)
        return 1.0 - ((2 * inter + self.smooth) / (card + self.smooth)).mean()

    def forward(self, pred_logits, target, corrupted):
        pred_logits = pred_logits.float()
        target, corrupted = target.float(), corrupted.float()

        hole = self._hole_mask(corrupted, target)
        w = 1.0 + hole * (self.hole_weight - 1.0)
        per_px = F.binary_cross_entropy_with_logits(pred_logits, target, reduction='none')
        L_bce = (per_px * w).mean()

        pred_lines = torch.sigmoid(-pred_logits)      # == 1 - sigmoid(x), no cancellation
        target_lines = (target < 0.5).float()         # binarized foreground for Dice

        r = self.gap_radius
        gap_region = F.max_pool2d(hole, 2 * r + 1, stride=1, padding=r)

        L_dice = self._dice(pred_lines, target_lines)
        L_gap_dice = self._dice(pred_lines, target_lines, gap_region)  # 0 if no holes

        total = (self.bce_weight * L_bce + self.dice_weight * L_dice
                 + self.gap_dice_weight * L_gap_dice)

        with torch.no_grad():
            hole_only = (per_px * hole).sum() / hole.sum().clamp(min=1.0)
        self.last_components = dict(
            bce=L_bce.item(), dice=L_dice.item(), gap_dice=L_gap_dice.item(),
            total=total.item(), hole_frac=hole.mean().item(), hole_only_bce=hole_only.item())
        return total
