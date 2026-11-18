"""Intro figures, one example per PDF (CODE BLOCK 15 of the notebook).

For each test file, saves a 1x4 figure (corrupted input | binarized prediction |
ground truth | error map) as its own PDF in --out-dir.

    python code/intro_figures.py --checkpoint /content/checkpoints/inkcloser_paper.pt
"""
import argparse
import os
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inkcloser.data import DEFAULT_SPLIT_ROOT, FrozenPairDataset, load_split  # noqa: E402
from inkcloser.metrics import line_stats, predict, summarize  # noqa: E402
from inkcloser.model import LineInpainter, load_weights  # noqa: E402
from inkcloser.viz import error_map  # noqa: E402

ERROR_LEGEND = [Patch(facecolor=(0, 0, 0), edgecolor="0.5", label="correct ink"),
                Patch(facecolor=(0, .7, 0), edgecolor="0.5", label="correctly filled gap"),
                Patch(facecolor=(.9, 0, 0), edgecolor="0.5", label="missed ink"),
                Patch(facecolor=(0, .3, 1), edgecolor="0.5", label="false ink")]


def intro_figure(model, device, dataset, fname, out_dir, thresh=0.5, show_titles=True,
                 show_legend=True, show_scores=False, fig_width_in=12):
    corr, clean, _ = dataset[dataset.files.index(fname)]
    c, g = corr[0].numpy(), clean[0].numpy()
    p = predict(model, corr[None], device)[0, 0].numpy()
    pb = (p >= thresh).astype(np.float32)          # 0 = ink, 1 = paper
    err, _ = error_map(c, pb, g)

    err_title = "Error map"
    if show_scores:
        sm = summarize(line_stats(c, pb, g, tol=0, regions=False))
        err_title += f" (P {sm['precision']:.0%}, R {sm['recall']:.0%})"

    panels = [(c, "Corrupted input"), (pb, "Prediction"), (g, "Ground truth"), (err, err_title)]
    with plt.rc_context({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 11}):
        fig, axes = plt.subplots(1, 4, figsize=(fig_width_in, fig_width_in / 4 + 0.6))
        for ax, (img, title) in zip(axes, panels):
            # interpolation="none" embeds the original pixels in the PDF (no resampling)
            ax.imshow(img, cmap=None if img.ndim == 3 else "gray",
                      vmin=0, vmax=1, interpolation="none")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_linewidth(0.6); s.set_color("0.4")
            if show_titles:
                ax.set_title(title)
        if show_legend:
            fig.legend(handles=ERROR_LEGEND, loc="lower center", ncol=4,
                       frameon=False, fontsize=10, bbox_to_anchor=(0.5, 0.0))
            plt.tight_layout(rect=(0, 0.08, 1, 1))
        else:
            plt.tight_layout()
        out = os.path.join(out_dir, os.path.splitext(fname)[0] + "_intro.pdf")
        fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split-root", default=DEFAULT_SPLIT_ROOT)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out-dir", default="/content/intro_figures")
    ap.add_argument("--files", nargs="*",
                    default=[f"pair_{i:05d}.png" for i in range(0, 100, 10)])
    ap.add_argument("--show-scores", action="store_true")
    ap.add_argument("--zip", action="store_true")
    args = ap.parse_args()

    split = load_split(args.split_root)
    dataset = FrozenPairDataset(split["test_clean_dir"], split["test_corr_dir"],
                                image_size=split["IMAGE_SIZE"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LineInpainter().to(device)
    load_weights(model, args.checkpoint, device)
    model.eval()

    shutil.rmtree(args.out_dir, ignore_errors=True)       # no stale PDFs
    os.makedirs(args.out_dir, exist_ok=True)
    available = set(dataset.files)
    missing = [f for f in args.files if f not in available]
    if missing:
        print(f"WARNING: not in the test set, skipped: {missing}")
    saved = [intro_figure(model, device, dataset, f, args.out_dir, show_scores=args.show_scores)
             for f in args.files if f in available]
    for s in saved:
        print("saved", s)
    assert saved, "no figures were generated"
    if args.zip:
        print("zip:", shutil.make_archive(args.out_dir, "zip", args.out_dir))


if __name__ == "__main__":
    main()
