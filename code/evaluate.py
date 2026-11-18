"""Paper results on the frozen test set (CODE BLOCK 14 of the notebook).

Evaluates the input (no restoration), morphological closing (k=5, 9) and a trained
InkCloser checkpoint on the same test pairs, then writes the main table (CSV +
LaTeX), the paired bootstrap tests, the gap-length table, per-image CSVs, figures
and summary.json to --results-dir.

Results computed elsewhere (e.g. the Sasaki et al. baseline, whose code is not in
this repo) can be merged into the same tables with --extra-results DIR, where DIR
holds a per_image_metrics.csv (and optionally gap_segments.csv) written by this
script or by the notebook. --from-csv skips inference and only re-aggregates.

Examples
  # full paper evaluation with the released weights (~1000 images)
  python code/evaluate.py --checkpoint /content/checkpoints/inkcloser_paper.pt

  # quick check on the first 40 test images, no figures
  python code/evaluate.py --checkpoint ... --max-images 40 --no-figures

  # rebuild the combined tables of the paper from the saved CSVs (no GPU)
  python code/evaluate.py --from-csv paper/v1/results paper/v1/results_sasaki \\
      --rename "Sasaki17 pretrained (scale 1)=Sasaki et al. (pretrained)" \\
      --rename "Sasaki17 retrained (scratch, mse, adadelta)=Sasaki et al. (retrained)"
"""
import argparse
import os
import platform
import shutil
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.amp import autocast
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inkcloser import paper_eval as pe  # noqa: E402
from inkcloser.data import DEFAULT_SPLIT_ROOT, FrozenPairDataset, load_split  # noqa: E402
from inkcloser.model import LineInpainter, load_weights  # noqa: E402

BASELINES = [pe.INPUT] + [f"Closing (k={k})" for k in pe.CLOSE_KS]


def read_results_dir(d, renames):
    df = pd.read_csv(os.path.join(d, "per_image_metrics.csv"))
    gp = os.path.join(d, "gap_segments.csv")
    gaps = pd.read_csv(gp) if os.path.exists(gp) else None
    df["method"] = df["method"].replace(renames)
    if gaps is not None:
        gaps["method"] = gaps["method"].replace(renames)
    return df, gaps


def merge_results(base_df, base_gaps, dirs, renames, files=None):
    """Add methods from other result folders; methods already present are kept as is."""
    df, gaps = base_df, base_gaps
    for d in dirs:
        x, xg = read_results_dir(d, renames)
        have = set(df["method"]) if df is not None else set()
        new = [m for m in x["method"].unique() if m not in have]
        if files is not None:
            x = x[x.file.isin(files)]
            xg = xg[xg.file.isin(files)] if xg is not None else None
        x = x[x.method.isin(new)]
        print(f"  {d}: adding {new}" + ("" if xg is not None else " (no gap_segments.csv)"))
        df = x if df is None else pd.concat([df, x], ignore_index=True)
        if xg is not None:
            # gap segments for any method that does not have them yet
            have_g = set(gaps["method"]) if gaps is not None else set()
            xg = xg[~xg.method.isin(have_g)]
            gaps = xg if gaps is None else pd.concat([gaps, xg], ignore_index=True)
    return df, gaps


def order_methods(present, ours):
    """Paper order: input, closing, any other baselines, ours last."""
    first = [m for m in BASELINES if m in present]
    rest = [m for m in present if m not in first and m != ours]
    return first + rest + ([ours] if ours in present else [])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split-root", default=DEFAULT_SPLIT_ROOT)
    ap.add_argument("--checkpoint", help="train.py / notebook checkpoint or bare state_dict")
    ap.add_argument("--name", default=pe.OURS, help="method name of the checkpoint in the tables")
    ap.add_argument("--results-dir", default="/content/paper_results")
    ap.add_argument("--max-images", type=int, default=None,
                    help="evaluate only the first N test images (check runs)")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--no-baselines", action="store_true", help="skip input / closing")
    ap.add_argument("--extra-results", nargs="*", default=[],
                    help="result folders whose methods are merged into the tables")
    ap.add_argument("--from-csv", nargs="*", default=None,
                    help="no inference: aggregate the per-image CSVs of these folders")
    ap.add_argument("--rename", action="append", default=[], metavar="OLD=NEW",
                    help="rename a method coming from --extra-results / --from-csv")
    ap.add_argument("--n-boot", type=int, default=pe.N_BOOT)
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--zip", action="store_true", help="also zip the results folder")
    args = ap.parse_args()

    renames = dict(r.split("=", 1) for r in args.rename)
    os.makedirs(args.results_dir, exist_ok=True)
    t_start = time.perf_counter()
    figures = not args.no_figures
    extra = dict()

    if args.from_csv is not None:
        print("aggregating saved per-image results:")
        df, gaps_df = merge_results(None, None, args.from_csv, renames)
        methods = order_methods(list(df["method"].unique()), args.name)
        model = None
    else:
        assert args.checkpoint, "--checkpoint is required unless --from-csv is used"
        split = load_split(args.split_root)
        files = None
        if args.max_images:
            files = sorted(f for f in os.listdir(split["test_clean_dir"])
                           if f.lower().endswith(".png"))[:args.max_images]
        test_dataset = FrozenPairDataset(split["test_clean_dir"], split["test_corr_dir"],
                                         files=files, image_size=split["IMAGE_SIZE"])
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers, pin_memory=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = LineInpainter().to(device)
        load_weights(model, args.checkpoint, device)
        model.eval()                                     # forward returns probabilities
        print(f"loaded {args.checkpoint} on {device}")

        @torch.no_grad()
        def run_model(corr):
            with autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                return model(corr.to(device, non_blocking=True)).float().cpu()[:, 0].numpy()

        def methods_fn(corr):
            c = corr[:, 0].numpy()
            out = {}
            if not args.no_baselines:
                out[pe.INPUT] = c
                for k in pe.CLOSE_KS:
                    out[f"Closing (k={k})"] = np.stack([pe.closing_baseline(x, k) for x in c])
            out[args.name] = run_model(corr)
            return out

        n_methods = 1 + (0 if args.no_baselines else 1 + len(pe.CLOSE_KS))
        print(f"evaluating {len(test_dataset)} test images × {n_methods} methods ...")
        df, gaps_df = pe.evaluate_all(test_loader, methods_fn)
        if args.extra_results:
            print("merging extra results:")
            df, gaps_df = merge_results(df, gaps_df, args.extra_results, renames,
                                        files=set(test_dataset.files))
        methods = order_methods(list(df["method"].unique()), args.name)

        # cost: parameters, inference time, memory
        extra["params_M"] = sum(p.numel() for p in model.parameters()) / 1e6
        if device.type == "cuda":
            x = next(iter(test_loader))[0][:1].to(device)
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad(), autocast("cuda", dtype=torch.float16):
                for _ in range(10):
                    model(x)
                torch.cuda.synchronize(); t = time.perf_counter()
                for _ in range(50):
                    model(x)
                torch.cuda.synchronize()
            extra.update(ms_per_image=1000 * (time.perf_counter() - t) / 50,
                         peak_mem_gb=torch.cuda.max_memory_allocated() / 1e9,
                         gpu=torch.cuda.get_device_name(0))
        extra.update(image_size=split["IMAGE_SIZE"], checkpoint=args.checkpoint)

    # every method must cover the same images (paired bootstrap)
    counts = df.groupby("method")["file"].nunique()
    assert counts.nunique() == 1, f"methods cover different image sets:\n{counts}"

    print(f"methods: {methods}")
    point, boot, n_test, _ = pe.write_results(df, gaps_df, methods, args.results_dir,
                                              ours=args.name, n_boot=args.n_boot,
                                              figures=figures)

    # qualitative figures (need the model)
    if model is not None and figures:
        @torch.no_grad()
        def predict_file(f):
            corr, clean, _ = test_dataset[test_dataset.files.index(f)]
            return corr[0].numpy(), run_model(corr[None])[0], clean[0].numpy()

        shown = [f for f in split["sample_files"] if f in test_dataset.files] or test_dataset.files[:4]
        pe.qual_figure(shown, predict_file, args.results_dir, "fig_qualitative", ours=args.name)
        pe.qual_figure(shown, predict_file, args.results_dir, "fig_qualitative_zoom", zoom=True,
                       ours=args.name)
        worst = pe.worst_files(df, ours=args.name)
        if worst:
            pe.qual_figure(worst, predict_file, args.results_dir, "fig_failure_cases", zoom=True,
                           ours=args.name)

    ref = df[df.method == methods[-1]]
    summary = pe.write_summary(
        args.results_dir, point, methods,
        n_test_images=n_test, n_erased_segments=int(ref.n_gaps.sum()),
        mean_hole_fraction=float(ref.hole_frac.mean()), **extra,
        env=dict(torch=torch.__version__, python=platform.python_version()))

    print("\n=== SUMMARY ===")
    print(f"test set: {n_test} images, {summary['n_erased_segments']} erased segments, "
          f"mean hole fraction {summary['mean_hole_fraction']:.3%}")
    if "params_M" in extra:
        print(f"model: {extra['params_M']:.2f} M params"
              + (f", {extra['ms_per_image']:.1f} ms/image, {extra['peak_mem_gb']:.2f} GB on "
                 f"{extra['gpu']}" if "gpu" in extra else ""))
    if args.zip:
        print("zip:", shutil.make_archive(args.results_dir, "zip", args.results_dir))
    print(f"all results in {args.results_dir}  [{time.perf_counter() - t_start:.0f} s]")


if __name__ == "__main__":
    main()
