"""Training-time metrics (CODE BLOCK 11 and evaluate() of CODE BLOCK 12a)."""
import cv2
import numpy as np
import torch
from torch.amp import autocast

METRIC_TOL = 0   # pixel tolerance for the per-epoch monitoring metrics


def _dilate(mask, tol):
    if tol <= 0:
        return mask
    k = np.ones((2 * tol + 1, 2 * tol + 1), np.uint8)
    return cv2.dilate(mask.astype(np.uint8), k).astype(bool)


def _count_regions(img_bool_ink, min_area=20):
    """Number of enclosed white regions. 4-connectivity on background pairs
       correctly with 8-connected ink (diagonal strokes still separate)."""
    bg = (~img_bool_ink).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bg, connectivity=4)
    # label 0 = ink; drop specks so they don't inflate the count
    return int(sum(1 for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area))


def line_stats(corrupted, pred, clean, tol=1, min_area=20, regions=True):
    """Raw counts for one image. All inputs: HxW arrays in [0,1], ink = < 0.5.
    Counts are summed over images, then turned into scores with summarize()."""
    c_ink, p_ink, g_ink = corrupted < 0.5, pred < 0.5, clean < 0.5

    added = p_ink & ~c_ink            # ink the model drew
    gap = g_ink & ~c_ink              # ink that was missing

    good_added = added & _dilate(g_ink, tol)
    found_gap = gap & _dilate(added, tol)

    out = dict(n_added=int(added.sum()), good_added=int(good_added.sum()),
               n_gap=int(gap.sum()), found_gap=int(found_gap.sum()),
               n_input_ink=int(c_ink.sum()), kept_ink=int((c_ink & p_ink).sum()))

    if regions:
        r_gt = _count_regions(g_ink, min_area)
        r_pred = _count_regions(p_ink, min_area)
        r_corr = _count_regions(c_ink, min_area)
        out.update(r_gt=r_gt, r_pred=r_pred, r_corr=r_corr,
                   r_exact=int(r_pred == r_gt),
                   r_lost=max(r_gt - r_corr, 0),                  # regions the holes merged
                   r_recovered=max(min(r_pred, r_gt) - r_corr, 0),
                   n_images=1)
    return out


def summarize(t):
    precision = t['good_added'] / t['n_added'] if t['n_added'] > 0 else 0.0
    recall = t['found_gap'] / max(t['n_gap'], 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    out = dict(precision=precision, recall=recall, f1=f1)

    if 'n_input_ink' in t:
        out['ink_kept'] = t['kept_ink'] / max(t['n_input_ink'], 1)
    if 'r_gt' in t:
        out['region_acc'] = t['r_exact'] / max(t.get('n_images', 1), 1)
        out['region_recovery'] = t['r_recovered'] / max(t['r_lost'], 1)
        out['regions_per_img'] = (t['r_pred'] / max(t.get('n_images', 1), 1),
                                  t['r_gt'] / max(t.get('n_images', 1), 1))
    return out


def add_stats(total, s):
    for k, v in s.items():
        total[k] = total.get(k, 0) + v
    return total


@torch.no_grad()
def predict(model, corr, device):
    """(B,1,H,W) tensor -> (B,1,H,W) float32 CPU probabilities (eval mode, fp16 autocast)."""
    was_training = model.training
    model.eval()
    with autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        pred = model(corr.to(device, non_blocking=True))
    if was_training:
        model.train()
    return pred.float().cpu()


@torch.no_grad()
def evaluate(model, loader, device, tol=METRIC_TOL, min_area=20, regions=True, max_batches=None):
    """Per-epoch monitoring metrics on the test loader (same as the notebook)."""
    total = {}
    tp = fp = fn = 0
    for b, (corr, clean, _) in enumerate(loader):
        pred = predict(model, corr, device)

        p_ink, g_ink = pred < 0.5, clean < 0.5
        tp += (p_ink & g_ink).sum().item()
        fp += (p_ink & ~g_ink).sum().item()
        fn += (~p_ink & g_ink).sum().item()

        for c, p, g in zip(corr[:, 0].numpy(), pred[:, 0].numpy(), clean[:, 0].numpy()):
            add_stats(total, line_stats(c, p, g, tol=tol, min_area=min_area, regions=regions))

        if max_batches and b + 1 >= max_batches:
            break

    s = summarize(total)
    px_p, px_r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    out = dict(
        fill_precision=s['precision'],    # added ink that lands on real lines (±tol px)
        fill_recall=s['recall'],          # erased ink that got redrawn (±tol px)
        fill_f1=s['f1'],
        ink_kept=s['ink_kept'],           # given ink the model preserved
        px_f1=2 * px_p * px_r / max(px_p + px_r, 1e-9),
        ink_ratio=(tp + fp) / max(tp + fn, 1),   # >1 = lines too thick
    )
    if regions:
        rp, rg = s['regions_per_img']
        out.update(region_acc=s['region_acc'],
                   region_recovery=s['region_recovery'],
                   regions_pred=rp,
                   regions_gt=rg)
    return out
