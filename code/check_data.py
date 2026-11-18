"""Sanity checks of the data loaders (CODE BLOCK 5 of the notebook).

Prints split integrity, shapes, pair consistency and train/test hole statistics,
and writes the histogram and preview figures to --out-dir.

    python code/check_data.py [--split-root /content/split] [--out-dir /content/data_check]
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inkcloser.data import (DEFAULT_SPLIT_ROOT, build_datasets, build_loaders,  # noqa: E402
                            load_split, preview_tensors)


def print_stats(name, img):
    print(f"{name:20s} min={img.min():.3f} max={img.max():.3f} "
          f"mean={img.mean():.3f} std={img.std():.3f} "
          f"ink_frac={(img < 0.5).float().mean():.4f}")


def pair_violations(corr, clean, tol=1 / 255):
    """Corruption only paints white over ink -> corrupted >= clean everywhere."""
    return (corr < clean - tol).float().mean().item()


def hole_frac(corr, clean):
    """Same definition as LineLoss._hole_mask: erased = white in input, ink in target."""
    return ((corr > 0.9) & (clean < 0.5)).float().mean(dim=(1, 2, 3))


@torch.no_grad()
def collect(loader, n_batches):
    holes, inks = [], []
    for k, batch in enumerate(loader):
        corr, clean = batch[0], batch[1]
        holes.append(hole_frac(corr, clean))
        inks.append((clean < 0.5).float().mean(dim=(1, 2, 3)))
        if k + 1 >= n_batches:
            break
    return torch.cat(holes).numpy(), torch.cat(inks).numpy()


def show_gray(ax, tensor, title):
    ax.imshow(tensor[0].numpy(), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(title, fontsize=9); ax.axis("off")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split-root", default=DEFAULT_SPLIT_ROOT)
    ap.add_argument("--out-dir", default="/content/data_check")
    ap.add_argument("--n-batches", type=int, default=25, help="batches for the distribution check")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(0)

    split = load_split(args.split_root)
    size = split["IMAGE_SIZE"]
    train_dataset, test_dataset, sample_dataset = build_datasets(split)
    train_loader, test_loader = build_loaders(train_dataset, test_dataset)
    ex_corrupted, ex_clean, example_files = preview_tensors(sample_dataset)

    # 0. split integrity
    print(f"train  : {len(train_dataset):5d} images")
    print(f"test   : {len(test_dataset):5d} pairs")
    print(f"sample : {len(sample_dataset):5d} pairs -> {example_files}")
    test_names = set(test_dataset.files)
    overlap = len(set(train_dataset.files) & test_names)
    print("train/test overlap:", overlap)
    print("sample inside test:", set(sample_dataset.files) <= test_names)
    assert overlap == 0, "train and test overlap"

    # 1. shapes / dtypes / range
    corrupted_batch, clean_batch = next(iter(train_loader))
    print("\n[train] corrupted:", tuple(corrupted_batch.shape), corrupted_batch.dtype)
    print("[train] clean    :", tuple(clean_batch.shape), clean_batch.dtype)
    t_corr, t_clean, t_names = next(iter(test_loader))
    print("[test ] corrupted:", tuple(t_corr.shape), "| files:", list(t_names))
    assert corrupted_batch.shape[1] == 1, "Expected 1 channel"
    assert corrupted_batch.shape[-2:] == (size, size)
    assert corrupted_batch.shape == clean_batch.shape
    assert 0.0 <= corrupted_batch.min() and corrupted_batch.max() <= 1.0

    print()
    print_stats("train clean", clean_batch[0])
    print_stats("train corrupted", corrupted_batch[0])
    print_stats("test clean", t_clean[0])
    print_stats("test corrupted", t_corr[0])
    u = torch.unique((clean_batch[0] * 255).round().int())
    print(f"\nclean unique values: {len(u)} (first 20) -> {u[:20].tolist()}")

    # 2. pair consistency
    print(f"\n[train] pixels darker in corrupted than clean: {pair_violations(corrupted_batch, clean_batch):.6f}")
    print(f"[test ] pixels darker in corrupted than clean: {pair_violations(t_corr, t_clean):.6f}")
    print("[train] hole_frac per image:", [f"{v:.4f}" for v in hole_frac(corrupted_batch, clean_batch)])
    print("[test ] hole_frac per image:", [f"{v:.4f}" for v in hole_frac(t_corr, t_clean)])

    # 3. train vs test distribution
    train_holes, train_ink = collect(train_loader, args.n_batches)
    test_holes, test_ink = collect(test_loader, args.n_batches)
    print(f"\nhole_frac  train {train_holes.mean():.4f} ± {train_holes.std():.4f}"
          f"   |   test {test_holes.mean():.4f} ± {test_holes.std():.4f}")
    print(f"ink_frac   train {train_ink.mean():.4f} ± {train_ink.std():.4f}"
          f"   |   test {test_ink.mean():.4f} ± {test_ink.std():.4f}")
    print(f"empty-hole images in train sample: {(train_holes < 1e-6).sum()} / {len(train_holes)}")

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
    bins = np.linspace(0, max(train_holes.max(), test_holes.max()) * 1.05, 30)
    ax[0].hist(train_holes, bins=bins, alpha=.6, label="train", density=True)
    ax[0].hist(test_holes, bins=bins, alpha=.6, label="test", density=True)
    ax[0].set_title("hole fraction"); ax[0].legend()
    ax[1].hist(train_ink, bins=30, alpha=.6, label="train", density=True)
    ax[1].hist(test_ink, bins=30, alpha=.6, label="test", density=True)
    ax[1].set_title("ink fraction"); ax[1].legend()
    plt.tight_layout(); fig.savefig(os.path.join(args.out_dir, "distributions.png"), dpi=100)
    plt.close(fig)

    # 4. visual check: the 4 fixed preview images
    n = len(sample_dataset)
    fig, axes = plt.subplots(n, 3, figsize=(13, 4.4 * n))
    for r in range(n):
        corr, clean = ex_corrupted[r], ex_clean[r]
        holes = ((corr > 0.9) & (clean < 0.5)).float()
        show_gray(axes[r, 0], clean, f"{example_files[r]} - clean")
        show_gray(axes[r, 1], corr, "corrupted (frozen)")
        ov = np.repeat(clean[0].numpy()[..., None], 3, axis=2)
        ov[holes[0].numpy() > 0] = (1.0, 0.0, 0.0)          # red = erased ink
        axes[r, 2].imshow(ov, interpolation="nearest")
        axes[r, 2].set_title(f"holes ({holes.mean():.3%} of pixels)", fontsize=9)
        axes[r, 2].axis("off")
    plt.tight_layout(); fig.savefig(os.path.join(args.out_dir, "previews.png"), dpi=80)
    plt.close(fig)
    print(f"\nfigures in {args.out_dir}")


if __name__ == "__main__":
    main()
