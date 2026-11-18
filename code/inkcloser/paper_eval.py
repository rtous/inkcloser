"""Paper results (CODE BLOCK 14 of the notebook), as reusable functions.

Metric families
  Fill      : precision / recall / F1 of the ink the method ADDS (tol 0 and 2 px)
  Pixel     : IoU / F1 of all ink, ink kept, ink ratio (>1 = lines too thick)
  Regions   : exact region count, merged regions re-separated
  Bucket    : each GT closed region is matched to the best predicted region (IoU).
              "Bucket acc." = GT regions matched with IoU >= BUCKET_IOU. A "leak" =
              one predicted region covering >= 50% of two GT regions.
  Gaps      : each erased segment is "closed" if >= GAP_CLOSED of its pixels are
              redrawn; reported overall and by gap length.

Ratios are pooled over all test images (micro average); 95% CIs come from a
bootstrap over images, with the same resamples for every method (paired).
"""
import json
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from .metrics import line_stats  # noqa: E402
from .viz import error_map  # noqa: E402

# configuration (identical to the notebook)
THRESH = 0.5        # ink = value < THRESH
MIN_AREA = 20       # ignore white regions smaller than this
TOLS = (0, 2)       # pixel tolerances for fill precision / recall
BUCKET_IOU = 0.9    # GT region is correctly paintable if matched with IoU >= this
CLOSE_KS = (5, 9)   # kernel sizes (px) of the morphological-closing baselines
GAP_CLOSED = 0.8    # an erased segment is closed if >= 80% of its pixels are redrawn
GAP_MIN_AREA = 4    # ignore erased "segments" smaller than this
GAP_BINS = (0, 8, 16, 32, 64, np.inf)   # gap length (px, longest bbox side)
N_BOOT = 1000
SEED = 0
OURS = "Ours"
INPUT = "Input (no restoration)"


# helpers
def div(a, b):
    """Elementwise a / b, NaN where b == 0 (undefined, shown as '—')."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    return np.where(b > 0, a / np.where(b > 0, b, 1.0), np.nan)


def f1(p, r):
    s = p + r
    return np.where(s > 0, 2 * p * r / np.where(s > 0, s, 1.0), 0.0) * np.where(np.isnan(s), np.nan, 1.0)


def closing_baseline(corr, k):
    """Classical gap closing: morphological closing of the ink mask."""
    ink = (corr < THRESH).astype(np.uint8)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return 1.0 - cv2.morphologyEx(ink, cv2.MORPH_CLOSE, ker).astype(np.float32)


def _bg_labels(ink):
    # 4-connected white regions (pairs with 8-connected ink); label 0 = ink pixels
    n, lab, st, _ = cv2.connectedComponentsWithStats((~ink).astype(np.uint8), connectivity=4)
    return n, lab, st[:, cv2.CC_STAT_AREA]


def region_stats(g_ink, p_ink):
    """Paint-bucket metrics: best-IoU matching of GT regions and leak count."""
    ng, lg, ag = _bg_labels(g_ink)
    npr, lp, ap = _bg_labels(p_ink)
    codes, inter = np.unique(lg.astype(np.int64).ravel() * npr + lp.ravel(), return_counts=True)
    gi, pj = codes // npr, codes % npr
    ok = (gi > 0) & (pj > 0) & (ag[gi] >= MIN_AREA)
    gi, pj, inter = gi[ok], pj[ok], inter[ok]

    valid_g = np.zeros(ng, bool); valid_g[1:] = ag[1:] >= MIN_AREA
    best = np.zeros(ng)
    if len(gi):
        iou = inter / (ag[gi] + ap[pj] - inter)
        np.maximum.at(best, gi, iou)
    best = best[valid_g]

    # leak: a predicted region that holds the majority of >= 2 GT regions
    maj = inter >= 0.5 * ag[gi]
    per_pred = np.bincount(pj[maj], minlength=npr) if maj.any() else np.zeros(1, int)
    n_leaks = int(np.maximum(per_pred - 1, 0).sum())
    return dict(n_gt_regions=int(valid_g.sum()),
                bucket_ok=int((best >= BUCKET_IOU).sum()),
                sum_best_iou=float(best.sum()),
                n_leaks=n_leaks, leak_free=int(n_leaks == 0))


def gap_components(c_ink, g_ink):
    """Erased segments of the test image (method independent)."""
    gap = (g_ink & ~c_ink).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(gap, connectivity=8)
    area = st[:, cv2.CC_STAT_AREA]
    length = np.maximum(st[:, cv2.CC_STAT_WIDTH], st[:, cv2.CC_STAT_HEIGHT])
    keep = np.zeros(n, bool); keep[1:] = area[1:] >= GAP_MIN_AREA
    return n, lab, area, length, keep


def gap_recall(p_ink, comps):
    n, lab, area, length, keep = comps
    found = np.bincount(lab.ravel(), weights=p_ink.ravel().astype(float), minlength=n)
    rec = found[keep] / area[keep]
    return length[keep], rec


# metric definitions (computed from pooled column sums)
# (name, fn(S) -> value, direction: up / down / one, format: pct / num)
def _P(S, t): return div(S[f"good_added_t{t}"], S[f"n_added_t{t}"])
def _R(S, t): return div(S[f"found_gap_t{t}"], S[f"n_gap_t{t}"])


METRICS = []
for _t in TOLS:
    METRICS += [
        (f"Fill P (τ={_t})", lambda S, t=_t: _P(S, t), "up", "pct"),
        (f"Fill R (τ={_t})", lambda S, t=_t: _R(S, t), "up", "pct"),
        (f"Fill F1 (τ={_t})", lambda S, t=_t: f1(_P(S, t), _R(S, t)), "up", "pct"),
    ]
METRICS += [
    ("Pixel IoU", lambda S: div(S["tp"], S["tp"] + S["fp"] + S["fn"]), "up", "pct"),
    ("Pixel F1", lambda S: div(2 * S["tp"], 2 * S["tp"] + S["fp"] + S["fn"]), "up", "pct"),
    ("Ink kept", lambda S: div(S["kept_ink"], S["n_input_ink"]), "up", "pct"),
    ("Ink ratio", lambda S: div(S["tp"] + S["fp"], S["tp"] + S["fn"]), "one", "num"),
    ("Gap closure", lambda S: div(S["gaps_closed"], S["n_gaps"]), "up", "pct"),
    ("Region count acc.", lambda S: div(S["r_exact"], S["n_images"]), "up", "pct"),
    ("Region recovery", lambda S: div(S["r_recovered"], S["r_lost"]), "up", "pct"),
    (f"Bucket acc. (IoU≥{BUCKET_IOU})", lambda S: div(S["bucket_ok"], S["n_gt_regions"]), "up", "pct"),
    ("Mean region IoU", lambda S: div(S["sum_best_iou"], S["n_gt_regions"]), "up", "pct"),
    ("Leaks / image", lambda S: div(S["n_leaks"], S["n_images"]), "down", "num"),
    ("Leak-free images", lambda S: div(S["leak_free"], S["n_images"]), "up", "pct"),
]
KEY_METRICS = [f"Fill F1 (τ={TOLS[0]})", "Gap closure",
               f"Bucket acc. (IoU≥{BUCKET_IOU})", "Leak-free images"]
MAIN_COLS = [f"Fill P (τ={TOLS[0]})", f"Fill R (τ={TOLS[0]})", f"Fill F1 (τ={TOLS[0]})",
             f"Fill F1 (τ={TOLS[-1]})", "Pixel F1", "Ink ratio", "Gap closure",
             f"Bucket acc. (IoU≥{BUCKET_IOU})", "Leaks / image", "Leak-free images"]
_SPEC = {name: (d, k) for name, _, d, k in METRICS}


def per_image_record(c, p, g, comps):
    c_ink, p_ink, g_ink = c < THRESH, p < THRESH, g < THRESH
    rec = dict(tp=int((p_ink & g_ink).sum()), fp=int((p_ink & ~g_ink).sum()),
               fn=int((~p_ink & g_ink).sum()))
    for t in TOLS:
        s = line_stats(c, p, g, tol=t, min_area=MIN_AREA, regions=(t == TOLS[0]))
        for k in ("n_added", "good_added", "n_gap", "found_gap"):
            rec[f"{k}_t{t}"] = s[k]
        if t == TOLS[0]:
            rec.update({k: s[k] for k in ("n_input_ink", "kept_ink", "r_gt", "r_pred",
                                          "r_corr", "r_exact", "r_lost", "r_recovered",
                                          "n_images")})
    rec.update(region_stats(g_ink, p_ink))
    lengths, rec_g = gap_recall(p_ink, comps)
    rec.update(n_gaps=len(rec_g), gaps_closed=int((rec_g >= GAP_CLOSED).sum()))
    return rec, lengths, rec_g


def evaluate_all(loader, methods_fn, log_every=25):
    """methods_fn(corr_batch) -> {method: (B, H, W) float arrays}.

    Returns (per-image DataFrame, per-gap-segment DataFrame)."""
    rows, gaps = [], []
    n_done, n_total = 0, len(loader.dataset)
    for bi, (corr, clean, names) in enumerate(loader):
        preds = methods_fn(corr)
        for b, fname in enumerate(names):
            c, g = corr[b, 0].numpy(), clean[b, 0].numpy()
            comps = gap_components(c < THRESH, g < THRESH)
            hole = float(((c > 0.9) & (g < 0.5)).mean())
            for m, P in preds.items():
                rec, lengths, rec_g = per_image_record(c, P[b], g, comps)
                rows.append(dict(method=m, file=fname, hole_frac=hole, **rec))
                gaps.append(pd.DataFrame(dict(method=m, file=fname, length=lengths, recall=rec_g)))
        n_done += len(names)
        if log_every and (bi + 1) % log_every == 0:
            print(f"  {n_done}/{n_total} images", flush=True)
    return pd.DataFrame(rows), pd.concat(gaps, ignore_index=True)


def aggregate(df, methods, rng, n_boot=N_BOOT):
    """Point estimates + bootstrap CIs; the same resamples for every method (paired)."""
    files = sorted(df["file"].unique())
    n = len(files)
    idx = rng.integers(0, n, (n_boot, n))
    cols = [c for c in df.columns if c not in ("method", "file", "hole_frac")]
    point, boot = {}, {}
    for m in methods:
        X = df[df["method"] == m].set_index("file").loc[files, cols]
        S_full = {c: X[c].to_numpy(float).sum() for c in cols}
        S_boot = {c: X[c].to_numpy(float)[idx].sum(1) for c in cols}
        point[m] = {name: float(fn(S_full)) for name, fn, *_ in METRICS}
        boot[m] = {name: np.asarray(fn(S_boot), float) for name, fn, *_ in METRICS}
    return point, boot, n


def fmt(v, kind):
    if v is None or np.isnan(v):
        return "—"
    return f"{100 * v:.1f}" if kind == "pct" else f"{v:.3f}"


def _tex_name(n):
    return n.replace("τ", r"$\tau$").replace("≥", r"$\geq$")


def latex_table(point, boot, methods, metric_names, caption, label):
    arrows = {"up": r"$\uparrow$", "down": r"$\downarrow$", "one": r"$\to 1$"}

    def best_of(name):
        d, _ = _SPEC[name]
        vals = {m: point[m][name] for m in methods if not np.isnan(point[m][name])}
        if not vals:
            return None
        key = {"up": lambda m: vals[m], "down": lambda m: -vals[m],
               "one": lambda m: -abs(vals[m] - 1)}[d]
        return max(vals, key=key)

    head = " & ".join(["Method"] + [_tex_name(n) + " " + arrows[_SPEC[n][0]] for n in metric_names])
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             r"\resizebox{\linewidth}{!}{%",
             r"\begin{tabular}{l" + "c" * len(metric_names) + "}", r"\toprule",
             head + r" \\", r"\midrule"]
    # bold every method that ties with the best (as printed); no bold if all tie
    bold = {}
    for n in metric_names:
        b = best_of(n)
        shown = {m: fmt(point[m][n], _SPEC[n][1]) for m in methods}
        winners = {m for m in methods if b is not None and shown[m] == shown[b]}
        bold[n] = winners if len(winners) < len(methods) else set()
    for m in methods:
        cells = [m]
        for n in metric_names:
            d, k = _SPEC[n]
            v = point[m][n]
            lo, hi = np.nanpercentile(boot[m][n], [2.5, 97.5]) if not np.isnan(v) else (np.nan, np.nan)
            s = fmt(v, k)
            if s == "—":
                cells.append("--")
                continue
            if m in bold[n]:
                s = r"\textbf{" + s + "}"
            hw = (hi - lo) / 2 * (100 if k == "pct" else 1)
            s += r"{\scriptsize$\pm$" + (f"{hw:.1f}" if k == "pct" else f"{hw:.3f}") + "}"
            cells.append(s)
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}",
              r"\caption{" + caption + "}", r"\label{" + label + "}", r"\end{table}"]
    return "\n".join(lines)


def savefig(fig, results_dir, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(results_dir, f"{name}.{ext}"), dpi=300, bbox_inches="tight")


def write_results(df, gaps_df, methods, results_dir, ours=OURS, n_boot=N_BOOT, seed=SEED,
                  figures=True):
    """Tables (CSV + LaTeX), paired bootstrap, gap-length table and figures.

    `methods` is the row order; `ours` is compared against every other method.
    Returns (point, boot, n_test, by_len_wide)."""
    os.makedirs(results_dir, exist_ok=True)
    df = df.copy()
    df.to_csv(os.path.join(results_dir, "per_image_metrics.csv"), index=False)
    if gaps_df is not None:
        gaps_df.to_csv(os.path.join(results_dir, "gap_segments.csv"), index=False)

    rng = np.random.default_rng(seed)
    point, boot, n_test = aggregate(df, methods, rng, n_boot)

    # 1. main table
    rows = []
    for m in methods:
        row = {"Method": m}
        for name, _, d, k in METRICS:
            v = point[m][name]
            if np.isnan(v):
                row[name] = "—"
            else:
                lo, hi = np.nanpercentile(boot[m][name], [2.5, 97.5])
                row[name] = f"{fmt(v, k)} [{fmt(lo, k)}, {fmt(hi, k)}]"
        rows.append(row)
    table = pd.DataFrame(rows).set_index("Method")
    pd.DataFrame({m: point[m] for m in methods}).T.to_csv(
        os.path.join(results_dir, "main_table_values.csv"))
    table.to_csv(os.path.join(results_dir, "main_table_with_ci.csv"))
    print("\n=== MAIN RESULTS (% unless noted; [95% bootstrap CI]) ===")
    with pd.option_context("display.width", 250, "display.max_columns", 20):
        print(table.T.to_string())

    n_seg = int(df[df.method == ours].n_gaps.sum()) if ours in methods else int(df.n_gaps.max())
    tex = latex_table(point, boot, methods, MAIN_COLS,
                      caption=(f"Line restoration on the held-out test set ({n_test} images, "
                               f"{n_seg} erased segments). "
                               "Percentages unless noted; $\\pm$ is half the 95\\% bootstrap CI. "
                               f"$\\tau$: pixel tolerance. Bucket acc.: GT regions recovered with "
                               f"IoU$\\geq${BUCKET_IOU}. Leak: one predicted region spanning two GT regions."),
                      label="tab:main_results")
    arrows_t = {"up": r"$\uparrow$", "down": r"$\downarrow$", "one": r"$\to 1$"}
    body = []
    for name, _, d, k in METRICS:
        cells = [_tex_name(name) + " " + arrows_t[d]]
        for m in methods:
            v = point[m][name]
            cells.append("--" if np.isnan(v) else fmt(v, k))
        body.append(" & ".join(cells) + r" \\")
    tex_full = "\n".join(
        [r"\begin{table}[t]", r"\centering", r"\small",
         r"\begin{tabular}{l" + "c" * len(methods) + "}", r"\toprule",
         " & ".join(["Metric"] + list(methods)) + r" \\", r"\midrule", *body,
         r"\bottomrule", r"\end{tabular}",
         r"\caption{All metrics on the test set (percentages unless noted; 95\% CIs in "
         r"\texttt{main\_table\_with\_ci.csv}).}", r"\label{tab:all_results}", r"\end{table}"])
    with open(os.path.join(results_dir, "table_main.tex"), "w") as fh:
        fh.write(tex)
    with open(os.path.join(results_dir, "table_all.tex"), "w") as fh:
        fh.write(tex_full)

    # 2. paired significance: ours vs each other method
    sig = []
    if ours in methods:
        for m in methods:
            if m == ours:
                continue
            for name in KEY_METRICS:
                d = boot[ours][name] - boot[m][name]
                d = d[~np.isnan(d)]
                if len(d) == 0:
                    continue
                sig.append(dict(baseline=m, metric=name,
                                diff=point[ours][name] - point[m][name],
                                ci_lo=np.percentile(d, 2.5), ci_hi=np.percentile(d, 97.5),
                                p_one_sided=max((d <= 0).mean(), 1 / n_boot)))
    sig_df = pd.DataFrame(sig)
    sig_df.to_csv(os.path.join(results_dir, "paired_bootstrap_vs_baselines.csv"), index=False)
    if len(sig_df):
        print(f"\n=== {ours} − baseline (paired bootstrap; p = P[diff ≤ 0]) ===")
        print(sig_df.round(4).to_string(index=False))

    # 3. by gap length (only for methods whose gap segments are available)
    by_len_wide = None
    if gaps_df is not None and len(gaps_df):
        gl_methods = [m for m in methods if m in set(gaps_df["method"])]
        labels = [f"<{GAP_BINS[1]}"] + [f"{a}–{b}" for a, b in zip(GAP_BINS[1:-2], GAP_BINS[2:-1])] \
                 + [f"≥{GAP_BINS[-2]}"]
        g = gaps_df[gaps_df.method.isin(gl_methods)].copy()
        g["bin"] = pd.cut(g["length"], bins=list(GAP_BINS), labels=labels, right=False)
        g["closed"] = g["recall"] >= GAP_CLOSED
        by_len = (g.groupby(["method", "bin"], observed=False)
                   .agg(n=("closed", "size"), closure=("closed", "mean"),
                        mean_recall=("recall", "mean"))
                   .reset_index())
        by_len_wide = by_len.pivot(index="bin", columns="method", values="closure")[gl_methods] * 100
        by_len_wide.insert(0, "# gaps", by_len[by_len.method == gl_methods[0]].set_index("bin")["n"])
        by_len_wide.to_csv(os.path.join(results_dir, "gap_closure_by_length.csv"))
        print(f"\n=== Gap closure (%) by gap length in px (closed = ≥{GAP_CLOSED:.0%} redrawn) ===")
        print(by_len_wide.round(1).to_string())
        tex_len = by_len_wide[by_len_wide["# gaps"] > 0].rename(columns={"# gaps": r"\# gaps"})
        tex_len.index = [str(s).replace("≥", r"$\geq$").replace("<", "$<$").replace("–", "--")
                         for s in tex_len.index]
        tex_len.index.name = "Gap length (px)"
        tex_len.columns.name = None
        closed_pct = f"{GAP_CLOSED:.0%}".replace("%", r"\%")
        with open(os.path.join(results_dir, "table_gap_length.tex"), "w") as fh:
            fh.write(tex_len.to_latex(
                caption=(r"Gap closure rate (\%) by erased-segment length (longest side of its "
                         r"bounding box, px). A gap is closed if $\geq$" + closed_pct
                         + " of its pixels are redrawn."),
                label="tab:gap_length", na_rep="--", float_format="%.1f", escape=False))
        if figures:
            fig, ax = plt.subplots(figsize=(5.5, 3.6))
            for m in gl_methods:
                ax.plot(labels, by_len_wide[m].values, marker="o", label=m,
                        lw=2.5 if m == ours else 1.2)
            ax.set_xlabel("gap length (px)"); ax.set_ylabel("gaps closed (%)")
            ax.set_ylim(0, 100); ax.grid(alpha=.3); ax.legend(fontsize=8)
            plt.tight_layout(); savefig(fig, results_dir, "fig_gap_closure_by_length"); plt.close(fig)

    # 4. per-image distributions
    if figures:
        t0 = TOLS[0]
        df["fill_f1"] = f1(div(df[f"good_added_t{t0}"], df[f"n_added_t{t0}"]),
                           div(df[f"found_gap_t{t0}"], df[f"n_gap_t{t0}"]))
        df["bucket_acc"] = div(df["bucket_ok"], df["n_gt_regions"])
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
        for ax, col, title in ((axes[0], "fill_f1", f"per-image fill F1 (τ={t0})"),
                               (axes[1], "bucket_acc", f"per-image bucket acc. (IoU≥{BUCKET_IOU})")):
            data = [df[(df.method == m)][col].dropna().values for m in methods]
            ax.boxplot(data, showfliers=False)
            ax.set_xticks(range(1, len(methods) + 1))
            ax.set_xticklabels([m.replace(" (", "\n(") for m in methods], fontsize=8)
            ax.set_title(title); ax.set_ylim(0, 1.02); ax.grid(alpha=.3, axis="y")
        plt.tight_layout(); savefig(fig, results_dir, "fig_per_image_boxplots"); plt.close(fig)

    return point, boot, n_test, by_len_wide


# qualitative figures
ERROR_LEGEND = [Patch(color=(0, 0, 0), label="correct ink"),
                Patch(color=(0, .7, 0), label="correctly filled gap"),
                Patch(color=(.9, 0, 0), label="missed ink"),
                Patch(color=(0, .3, 1), label="false ink")]


def zoom_box(c, g, size=224):
    gap = ((g < THRESH) & ~(c < THRESH)).astype(np.uint8)
    n, _, st, cen = cv2.connectedComponentsWithStats(gap, connectivity=8)
    H, W = g.shape
    cx, cy = (W / 2, H / 2) if n <= 1 else cen[1 + np.argmax(st[1:, cv2.CC_STAT_AREA])]
    y0 = int(np.clip(cy - size / 2, 0, H - size)); x0 = int(np.clip(cx - size / 2, 0, W - size))
    return np.s_[y0:y0 + size, x0:x0 + size]


def qual_figure(files, predict_file, results_dir, name, zoom=False, ours=OURS):
    """predict_file(f) -> (corrupted, probability map, clean) HxW arrays."""
    k_show = max(CLOSE_KS)
    cols = ["Input", f"Closing (k={k_show})", ours, "Ground truth", "Error map (ours)"]
    fig, axes = plt.subplots(len(files), len(cols), figsize=(3.2 * len(cols), 3.3 * len(files)))
    axes = np.atleast_2d(axes)
    for r, f in enumerate(files):
        c, p, g = predict_file(f)
        pb = (p >= THRESH).astype(np.float32)
        err, _ = error_map(c, pb, g)
        imgs = [c, closing_baseline(c, k_show), pb, g, err]
        sl = zoom_box(c, g) if zoom else np.s_[:, :]
        for col, (img, title) in enumerate(zip(imgs, cols)):
            ax = axes[r, col]
            ax.imshow(img[sl], cmap=None if img.ndim == 3 else "gray",
                      vmin=0, vmax=1, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(title, fontsize=11)
        axes[r, 0].set_ylabel(f, fontsize=8)
    fig.legend(handles=ERROR_LEGEND, loc="lower center", ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, -0.01))
    plt.tight_layout(rect=(0, 0.03, 1, 1)); savefig(fig, results_dir, name); plt.close(fig)


def worst_files(df, ours=OURS, n=3):
    """Failure cases shown in the paper: lowest per-image fill F1 among images with >=200 gap px."""
    t0 = TOLS[0]
    o = df[df.method == ours].copy()
    o["fill_f1"] = f1(div(o[f"good_added_t{t0}"], o[f"n_added_t{t0}"]),
                      div(o[f"found_gap_t{t0}"], o[f"n_gap_t{t0}"]))
    return (o[o[f"n_gap_t{t0}"] >= 200]
            .sort_values(["fill_f1", "n_leaks"], ascending=[True, False])["file"].head(n).tolist())


def config_dict():
    return dict(threshold=THRESH, min_region_area=MIN_AREA, tolerances=TOLS,
                bucket_iou=BUCKET_IOU, gap_closed=GAP_CLOSED, gap_min_area=GAP_MIN_AREA,
                closing_kernels=CLOSE_KS, n_boot=N_BOOT, seed=SEED)


def write_summary(results_dir, point, methods, **extra):
    summary = dict(**extra, config=config_dict(),
                   results={m: {k: (None if np.isnan(v) else v) for k, v in point[m].items()}
                            for m in methods})
    with open(os.path.join(results_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    return summary
