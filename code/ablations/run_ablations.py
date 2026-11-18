"""Train and evaluate the ablation variants (EXP-1), then write the ablation tables.

All variants, including the full-model reference, are trained with the same
reduced schedule (--epochs-budget epochs of the paper's 30-epoch cosine schedule,
same seed, same data and augmentation) and evaluated on the full frozen test set
with the paper's metric code.

Stages (each skips work that is already done, so the script can be re-run after
an interruption):
  train     -> <ckpt-root>/<variant>/last.pt
  evaluate  -> <ckpt-root>/<variant>/per_image_metrics.csv (+ gap_segments.csv)
  report    -> <results-dir>/ (ablation_table.csv/.tex + the standard paper tables)

Examples
  python code/ablations/run_ablations.py                          # everything
  python code/ablations/run_ablations.py --variants full no_gated --stages train
  python code/ablations/run_ablations.py --stages report          # tables only
  # check run: 5 steps per variant, 8 test images
  python code/ablations/run_ablations.py --check --ckpt-root /tmp/abl --results-dir /tmp/abl_res
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.amp import autocast
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
REPO = os.path.dirname(CODE)
sys.path.insert(0, CODE)
sys.path.insert(0, HERE)
import train as train_script  # noqa: E402
from inkcloser import paper_eval as pe  # noqa: E402
from inkcloser.data import DEFAULT_SPLIT_ROOT, FrozenPairDataset, load_split  # noqa: E402
from inkcloser.model import LineInpainter, load_weights  # noqa: E402
from variants import VARIANTS, build_model  # noqa: E402

REFERENCE = "released"   # key of the optional row for the released paper model
TABLE_COLS = [f"Fill F1 (τ={pe.TOLS[0]})", f"Fill F1 (τ={pe.TOLS[-1]})", "Gap closure",
              f"Bucket acc. (IoU≥{pe.BUCKET_IOU})", "Leaks / image", "Leak-free images",
              "Ink kept", "Ink ratio"]


def is_trained(run_dir, args):
    p = os.path.join(run_dir, "last.pt")
    if not os.path.exists(p):
        return False
    ck = torch.load(p, map_location="cpu", weights_only=False)
    if args.check:
        return ck["global_step"] >= args.check_steps
    if args.max_steps:
        return ck["global_step"] >= args.max_steps
    return ck["epochs_done"] >= args.epochs_budget


def train_variant(key, args):
    run_dir = os.path.join(args.ckpt_root, key)
    if is_trained(run_dir, args):
        print(f"[train] {key}: already trained, skipping")
        return
    label, opts, loss_args = VARIANTS[key]
    print(f"\n[train] {key} ({label}) -> {run_dir}", flush=True)
    argv = ["--split-root", args.split_root, "--out-dir", run_dir,
            "--epochs", "30", "--stop-after", str(args.epochs_budget),
            "--seed", str(args.seed), "--eval-batches", "0", "--preview-every", "0",
            "--log-every", "50"] + loss_args
    last = os.path.join(run_dir, "last.pt")
    if os.path.exists(last):
        argv += ["--resume", last]
    if args.check:
        argv += ["--max-steps", str(args.check_steps), "--log-every", "1"]
    elif args.max_steps:
        argv[argv.index("--stop-after") + 1] = "30"
        argv += ["--max-steps", str(args.max_steps)]
    with open(os.path.join(os.makedirs(run_dir, exist_ok=True) or run_dir, "variant.json"), "w") as fh:
        json.dump(dict(key=key, label=label, model=opts, loss_args=loss_args), fh, indent=2)
    t0 = time.perf_counter()
    train_script.main(argv, model_fn=lambda: build_model(**opts))
    with open(os.path.join(run_dir, "train_time.json"), "w") as fh:
        json.dump(dict(minutes=(time.perf_counter() - t0) / 60), fh)


def speed(model, x, device):
    if device.type != "cuda":
        return {}
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad(), autocast("cuda", dtype=torch.float16):
        for _ in range(10):
            model(x)
        torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(50):
            model(x)
        torch.cuda.synchronize()
    return dict(ms_per_image=1000 * (time.perf_counter() - t) / 50,
                peak_mem_gb=torch.cuda.max_memory_allocated() / 1e9)


def evaluate_variant(key, args, loader, device):
    if key == REFERENCE:
        run_dir = os.path.join(args.ckpt_root, REFERENCE)
        label, ckpt, model = "InkCloser (released, paper)", args.reference_checkpoint, LineInpainter()
    else:
        run_dir = os.path.join(args.ckpt_root, key)
        label, opts, _ = VARIANTS[key]
        ckpt, model = os.path.join(run_dir, "last.pt"), build_model(**opts)
    os.makedirs(run_dir, exist_ok=True)
    out_csv = os.path.join(run_dir, "per_image_metrics.csv")
    info_path = os.path.join(run_dir, "eval_info.json")
    n_files = len(loader.dataset)
    if os.path.exists(out_csv) and os.path.exists(info_path):
        info = json.load(open(info_path))
        if info.get("n_images") == n_files:
            print(f"[eval] {key}: already evaluated, skipping")
            return
    if not os.path.exists(ckpt):
        print(f"[eval] {key}: no checkpoint {ckpt}, skipping")
        return
    print(f"\n[eval] {key} ({label}) on {n_files} images", flush=True)
    model = model.to(device)
    load_weights(model, ckpt, device)
    model.eval()

    @torch.no_grad()
    def methods_fn(corr):
        with autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            p = model(corr.to(device, non_blocking=True)).float().cpu()[:, 0].numpy()
        return {label: p}

    df, gaps = pe.evaluate_all(loader, methods_fn, log_every=50)
    df.to_csv(out_csv, index=False)
    gaps.to_csv(os.path.join(run_dir, "gap_segments.csv"), index=False)
    info = dict(key=key, label=label, checkpoint=ckpt, n_images=n_files,
                params_M=sum(p.numel() for p in model.parameters()) / 1e6,
                **speed(model, next(iter(loader))[0][:1].to(device), device))
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    info.update(train_steps=ck.get("global_step"), epochs_done=ck.get("epochs_done"))
    tt = os.path.join(run_dir, "train_time.json")
    if os.path.exists(tt):
        info["train_minutes"] = json.load(open(tt))["minutes"]
    with open(info_path, "w") as fh:
        json.dump(info, fh, indent=2)
    del model
    torch.cuda.empty_cache()


def report(keys, args):
    dfs, gaps, infos = [], [], []
    for key in keys:
        run_dir = os.path.join(args.ckpt_root, key)
        p = os.path.join(run_dir, "per_image_metrics.csv")
        if not os.path.exists(p):
            print(f"[report] {key}: not evaluated yet, left out")
            continue
        dfs.append(pd.read_csv(p))
        g = os.path.join(run_dir, "gap_segments.csv")
        if os.path.exists(g):
            gaps.append(pd.read_csv(g))
        infos.append(json.load(open(os.path.join(run_dir, "eval_info.json"))))
    if not dfs:
        print("[report] nothing to report")
        return
    df = pd.concat(dfs, ignore_index=True)
    gaps_df = pd.concat(gaps, ignore_index=True) if gaps else None
    labels = [i["label"] for i in infos]
    full = VARIANTS["full"][0]
    ours = full if full in labels else labels[0]
    # common image set (paired bootstrap)
    common = set.intersection(*(set(df[df.method == m].file) for m in labels))
    df = df[df.file.isin(common)]
    if gaps_df is not None:
        gaps_df = gaps_df[gaps_df.file.isin(common)]

    os.makedirs(args.results_dir, exist_ok=True)
    point, boot, n_test, _ = pe.write_results(df, gaps_df, labels, args.results_dir, ours=ours,
                                              figures=not args.check)

    # compact ablation table: key metrics ± CI half-width, ink kept, params, ms
    rows, tex_rows = [], []
    for info in infos:
        m = info["label"]
        row = {"Variant": m}
        cells = [m.replace("×", r"$\times$")]
        for name in TABLE_COLS:
            d, k = pe._SPEC[name]
            v = point[m][name]
            lo, hi = np.nanpercentile(boot[m][name], [2.5, 97.5])
            hw = (hi - lo) / 2 * (100 if k == "pct" else 1)
            row[name] = v
            row[name + " ±"] = hw / (100 if k == "pct" else 1)
            if name == "Ink kept":
                cells.append(f"{100 * v:.2f}")
            else:
                cells.append(pe.fmt(v, k) + r"{\scriptsize$\pm$" + (f"{hw:.1f}" if k == "pct" else f"{hw:.2f}") + "}")
        row.update(params_M=info["params_M"], ms_per_image=info.get("ms_per_image"),
                   train_steps=info.get("train_steps"), train_minutes=info.get("train_minutes"))
        cells += [f"{info['params_M']:.1f}",
                  f"{info['ms_per_image']:.0f}" if info.get("ms_per_image") else "--"]
        rows.append(row)
        tex_rows.append(" & ".join(cells) + r" \\")
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(args.results_dir, "ablation_table.csv"), index=False)

    arrows = {"up": r"$\uparrow$", "down": r"$\downarrow$", "one": r"$\to 1$"}
    head = ["Variant"] + [pe._tex_name(n) + " " + arrows[pe._SPEC[n][0]] for n in TABLE_COLS] \
           + ["Params (M)", "ms/img"]
    steps = sorted({i.get("train_steps") for i in infos if i["key"] != REFERENCE} - {None})
    tex = "\n".join([
        r"\begin{table}[t]", r"\centering", r"\small", r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{l" + "c" * (len(head) - 1) + "}", r"\toprule",
        " & ".join(head) + r" \\", r"\midrule", *tex_rows, r"\bottomrule", r"\end{tabular}}",
        r"\caption{Ablation study on the test set (" + str(n_test) + r" images). All variants "
        r"are trained with the same reduced schedule (" + ", ".join(map(str, steps)) +
        r" steps, batch 4, same seed). Percentages unless noted; $\pm$ is half the 95\% "
        r"bootstrap CI. Run time measured on a single image, fp16.}",
        r"\label{tab:ablation}", r"\end{table}"])
    with open(os.path.join(args.results_dir, "table_ablation.tex"), "w") as fh:
        fh.write(tex)
    with open(os.path.join(args.results_dir, "variants.json"), "w") as fh:
        json.dump(infos, fh, indent=2)

    print("\n=== ABLATION ===")
    show = tab[["Variant"] + TABLE_COLS + ["params_M", "ms_per_image"]].copy()
    for c in TABLE_COLS:
        if pe._SPEC[c][1] == "pct":
            show[c] = (100 * show[c]).round(2 if c == "Ink kept" else 1)
        else:
            show[c] = show[c].round(3)
    with pd.option_context("display.width", 250, "display.max_columns", 20):
        print(show.to_string(index=False))
    print(f"\nresults in {args.results_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS), choices=list(VARIANTS))
    ap.add_argument("--stages", nargs="*", default=["train", "evaluate", "report"],
                    choices=["train", "evaluate", "report"])
    ap.add_argument("--split-root", default=DEFAULT_SPLIT_ROOT)
    ap.add_argument("--ckpt-root", default="/content/checkpoints/ablations")
    ap.add_argument("--results-dir", default=os.path.join(REPO, "paper", "v1", "results_ablations"))
    ap.add_argument("--epochs-budget", type=int, default=1,
                    help="epochs per variant (of the paper's 30-epoch cosine schedule)")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="step budget per variant instead of --epochs-budget "
                         "(e.g. 9155 = the released model's training length)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reference-checkpoint", default="/content/checkpoints/inkcloser_paper.pt",
                    help="released paper model, added as a reference row ('' to skip)")
    ap.add_argument("--max-images", type=int, default=None, help="evaluate first N test images")
    ap.add_argument("--check", action="store_true", help="tiny run: few steps, 8 test images")
    ap.add_argument("--check-steps", type=int, default=5)
    args = ap.parse_args()
    if args.check and args.max_images is None:
        args.max_images = 8

    keys = list(args.variants)
    if "train" in args.stages:
        for key in keys:
            train_variant(key, args)

    eval_keys = keys + ([REFERENCE] if args.reference_checkpoint else [])
    if "evaluate" in args.stages:
        split = load_split(args.split_root)
        files = None
        if args.max_images:
            files = sorted(f for f in os.listdir(split["test_clean_dir"])
                           if f.lower().endswith(".png"))[:args.max_images]
        ds = FrozenPairDataset(split["test_clean_dir"], split["test_corr_dir"], files=files,
                               image_size=split["IMAGE_SIZE"])
        loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=2, pin_memory=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for key in eval_keys:
            evaluate_variant(key, args, loader, device)

    if "report" in args.stages:
        report(eval_keys, args)


if __name__ == "__main__":
    main()
