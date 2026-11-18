"""Error map, prediction previews and training curves (CODE BLOCKS 12a / 12b).

Figures are written to disk instead of being shown inline.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .metrics import METRIC_TOL, line_stats, predict, summarize  # noqa: E402

LOSS_KEYS = ("total", "bce", "dice", "gap_dice", "hole_only_bce")


def error_map(corr, pred_bin, gt):
    """White bg, black = correct ink, green = correctly filled gap,
       red = missed ink, blue = false ink."""
    c_ink, p_ink, g_ink = corr < 0.5, pred_bin < 0.5, gt < 0.5
    rgb = np.ones(corr.shape + (3,), np.float32)
    rgb[p_ink & g_ink] = (0.0, 0.0, 0.0)
    rgb[p_ink & g_ink & ~c_ink] = (0.0, 0.7, 0.0)
    rgb[g_ink & ~p_ink] = (0.9, 0.0, 0.0)
    rgb[p_ink & ~g_ink] = (0.0, 0.3, 1.0)
    gap = g_ink & ~c_ink
    fill_recall = (p_ink & gap).sum() / max(gap.sum(), 1)
    return rgb, fill_recall


def save_predictions(model, device, corrupted, clean, names, tag, out_dir):
    pred = predict(model, corrupted, device)
    binar = (pred > 0.5).float()

    n = corrupted.shape[0]
    fig, axes = plt.subplots(n, 5, figsize=(25, 5 * n))
    axes = np.atleast_2d(axes)
    for r in range(n):
        c, p, b, g = (t[r, 0].numpy() for t in (corrupted, pred, binar, clean))
        err, _ = error_map(c, b, g)
        st = line_stats(c, b, g, tol=METRIC_TOL)
        sm = summarize(st)
        err_title = (f"Errors | fill P {sm['precision']:.0%} R {sm['recall']:.0%}"
                     f" | regions {st['r_pred']}/{st['r_gt']}")
        panels = [(c, "Corrupted"), (p, "Prediction"), (b, "Binarized"),
                  (g, "Ground truth"), (err, err_title)]
        for col, (img, title) in enumerate(panels):
            ax = axes[r, col]
            if img.ndim == 2:
                ax.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            else:
                ax.imshow(img, interpolation="nearest")
            ax.set_title(f"{names[r]} - {title}" if col == 0 else title)
            ax.axis("off")
    fig.suptitle(str(tag), fontsize=18)
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, f"{tag}.png"), dpi=100)
    plt.close(fig)


def _smooth(y, w=50):
    y = np.asarray(y, float)
    if len(y) < w:
        return y
    return np.convolve(y, np.ones(w) / w, mode="valid")


def save_curves(loss_log, history, tag, out_dir, window=50):
    fig, axes = plt.subplots(1, 2, figsize=(16, 4.5))

    ax = axes[0]
    for k in LOSS_KEYS:
        y = loss_log.get(k, [])
        if not y:
            continue
        s = _smooth(y, window)
        x = np.arange(len(y) - len(s) + 1, len(y) + 1)   # align each average to its last step
        ax.plot(x, s, label=k)
    ax.set_yscale("log")
    ax.set_xlabel("step"); ax.set_title(f"train losses (moving avg {window})")
    ax.legend(); ax.grid(alpha=.3)

    ax = axes[1]
    if history:
        ep = np.arange(1, len(history) + 1)
        for k in ("fill_precision", "fill_recall", "fill_f1", "region_acc", "ink_kept"):
            if k in history[0]:
                ax.plot(ep, [h[k] for h in history], marker="o", label=k)
        ax.set_ylim(0, 1); ax.legend()
    else:
        ax.text(0.5, 0.5, "test metrics appear after epoch 1",
                ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("epoch"); ax.set_title("test metrics"); ax.grid(alpha=.3)

    fig.suptitle(str(tag), fontsize=14)
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, f"curves_{tag}.png"), dpi=100)
    plt.close(fig)
