# Ablations (EXP-1 of the response to reviewers)

| key | variant | what changes |
|---|---|---|
| `full` | InkCloser (full) | reference, same reduced schedule as the others |
| `no_attn28` | w/o 28×28 attention | bottleneck transformer (6 blocks) removed |
| `no_branch56` | w/o 56×56 branch | dilated stack + 56×56 transformer (3 blocks) removed |
| `no_gated` | w/o gated conv | every encoder gated conv → standard conv |
| `no_gap_dice` | w/o local (gap) Dice | β_gap = 0 |
| `plain_unet_ours_loss` | Plain U-Net + our loss | no gates, no attention, no dilated stack |
| `plain_unet_bce` | Plain U-Net + BCE | same network, unweighted BCE only (λh = 1, β_dice = β_gap = 0) |
| `released` | InkCloser (released, paper) | the published weights (9,155 steps), reference row only |

Removed modules are replaced by `nn.Identity`, so the rest of the U-Net (encoder,
skip connections, decoder, head) is unchanged. Every variant is trained with the
paper's settings (optimizer, learning rates, augmentation, seed 0) for the same reduced
budget: 1 epoch = 2,250 steps of the paper's 30-epoch cosine schedule.

```bash
python code/ablations/run_ablations.py            # train + evaluate + report (~4 h on a T4)
python code/ablations/run_ablations.py --stages report
python code/ablations/run_ablations.py --check --ckpt-root /tmp/abl --results-dir /tmp/abl_res
```

Checkpoints and per-variant per-image metrics go to `/content/checkpoints/ablations/<key>/`;
tables go to `paper/v1/results_ablations/` (`table_ablation.tex`, `ablation_table.csv`,
and the standard paper tables / paired bootstrap against the full model). The script
skips finished stages, so rerun it after an interruption.

## Full-length runs (9,155 steps) and resuming after a disconnect

The 1-epoch budget left the 28×28 attention question open (see Results), so `full` and
`no_attn28` are retrained for the released model's length, 9,155 steps (4 epochs + 155
steps of the 5th, same 30-epoch cosine schedule and seed). `--max-steps` replaces
`--epochs-budget`; use a separate checkpoint root and results dir so the 1-epoch runs
are not overwritten. About 5.5 h on a T4 (≈2.8 h `full`, ≈2.3 h `no_attn28`, plus eval).

```bash
python code/ablations/run_ablations.py --variants full no_attn28 --max-steps 9155 \
    --ckpt-root /content/checkpoints/ablations_9155 \
    --results-dir paper/v1/results_ablations_9155
```

The `released` reference row is added automatically (don't pass it in `--variants`).

**Surviving a Colab reset.** A VM reset wipes `/content` (dataset, checkpoints, released
weights). With Google Drive mounted, mirror the checkpoints to
`MyDrive/inkrepair/tmp/ablations_9155/` while training runs:

```bash
nohup code/ablations/sync_to_drive.sh /content/checkpoints/ablations_9155 \
    /content/drive/MyDrive/inkrepair/tmp/ablations_9155 > /content/sync.log 2>&1 &
```

`train.py` writes `last.pt` only at epoch ends (every 2,250 steps, ~41 min for `full`),
so a disconnect loses at most the current epoch plus up to 5 min of sync lag.

To resume in a fresh VM:

1. Mount Drive in the notebook (`from google.colab import drive; drive.mount('/content/drive')`).
2. `python code/download_dataset.py` (→ `/content/split`) and
   `python code/download_checkpoint.py` (→ `/content/checkpoints/inkcloser_paper.pt`).
3. Rerun the command above with `--ckpt-root /content/drive/MyDrive/inkrepair/tmp/ablations_9155`.
   The runner resumes each variant from its `last.pt` and skips finished train/eval stages.
   (Training straight onto Drive is fine for resuming; it just writes ~0.5 GB per epoch there.)
4. Check progress in `<ckpt-root>/run.log` if the run was launched with
   `nohup ... > <ckpt-root>/run.log 2>&1 &`.

Note: a resumed run's `train_time.json` only covers the last session.

## Results

Run on a Tesla T4 (Colab), 1,000 test images. All rows except the last are trained for
2,250 steps (1 epoch, batch 4, seed 0). Percentages except leaks/image and params; ± in
`table_ablation.tex` is half the 95% bootstrap CI. Train time is for the 2,250 steps.

| Variant | Params | Fill F1 τ=0 | Fill F1 τ=2 | Gap closure | Bucket acc. | Leaks/img ↓ | Leak-free | ms/img | Train |
|---|---|---|---|---|---|---|---|---|---|
| InkCloser (full) | 43.4M | 76.6 | 91.5 | 42.2 | 90.0 | 3.49 | 26.9 | 98 | 43 min |
| w/o 28×28 attention | 24.4M | 76.4 | 91.9 | 44.1 | 91.3 | 2.81 | 32.4 | 82 | 35 min |
| w/o 56×56 branch | 35.1M | 76.3 | 91.8 | 38.8 | 90.5 | 3.31 | 26.4 | 71 | 27 min |
| w/o gated conv | 37.4M | 69.5 | 86.3 | 41.2 | 87.5 | 4.26 | 21.2 | 88 | 38 min |
| w/o local (gap) Dice | 43.4M | 71.6 | 87.4 | 31.0 | 85.0 | 5.60 | 14.7 | 98 | 43 min |
| Plain U-Net + our loss | 10.2M | 67.9 | 85.9 | 27.3 | 84.8 | 5.52 | 11.5 | 46 | 16 min |
| Plain U-Net + BCE | 10.2M | 66.1 | 85.5 | 18.6 | 77.8 | 8.71 | 10.4 | 46 | 16 min |
| *InkCloser (released, 9,155 steps)* | 43.4M | 78.5 | 92.8 | 70.5 | 94.7 | 1.47 | 45.0 | 98 | – |

Paired bootstrap against `full` (1,000 resamples, `paired_bootstrap_vs_baselines.csv`):

- **Local (gap) Dice** is the most important single component: removing it costs
  11.2 points of gap closure, 5.0 of bucket accuracy and 12.2 of leak-free images.
- **Gated convolutions** mainly drive fill precision: −7.1 points of fill F1 (τ=0) and
  −2.5 of bucket accuracy without them; gap closure −1.0.
- **56×56 branch** adds 3.4 points of gap closure; its other metrics are within noise.
- **Loss vs. architecture**: on a plain U-Net our loss alone raises gap closure from 18.6%
  to 27.3% and bucket accuracy from 77.8% to 84.8%; the full architecture then reaches 42.2%.
- **28×28 bottleneck attention** shows no benefit at this budget: the variant without it
  is slightly but significantly better on gap closure (+1.9), bucket accuracy (+1.3) and
  leak-free images (+5.5). A plausible (untested) explanation is that the transformer
  blocks need longer training than 2,250 steps; retraining `full` and `no_attn28` for
  9,155 steps would settle it (about 5 h on a T4).
- The released model, trained ~4× longer, is far ahead of the 1-epoch `full` on the
  structural metrics (gap closure 70.5 vs 42.2), so absolute numbers here are not
  comparable to the paper tables; only the differences between rows are.

### Full-length results (9,155 steps)

`full` and `no_attn28` retrained for 9,155 steps (167 and 139 min on a T4), 1,000 test
images, tables in `paper/v1/results_ablations_9155/`.
In the paper: Table `tab:ablation_long` and the paragraph "Bottleneck attention under
longer training" at the end of the Ablation Study section (the paper gives no step count;
see `paper/v1/HIDDEN_ISSUES.txt`, D.12b).

| Variant | Params | Fill F1 τ=0 | Fill F1 τ=2 | Gap closure | Bucket acc. | Leaks/img ↓ | Leak-free | ms/img |
|---|---|---|---|---|---|---|---|---|
| InkCloser (full) | 43.4M | 80.9 | 94.1 | 59.1 | 94.2 | 1.81 | 42.4 | 90 |
| w/o 28×28 attention | 24.4M | 80.2 | 93.7 | 62.6 | 94.0 | 1.88 | 41.2 | 77 |
| *InkCloser (released)* | 43.4M | 78.5 | 92.8 | 70.5 | 94.7 | 1.47 | 45.0 | 90 |

Paired bootstrap, full − no_attn28 (95% CI):

| Metric | 1 epoch (2,250 steps) | 9,155 steps |
|---|---|---|
| Fill F1 (τ=0) | +0.1 [−0.1, +0.3] | **+0.7** [+0.4, +1.0] |
| Gap closure | **−1.9** [−2.5, −1.3] | **−3.6** [−4.2, −3.1] |
| Bucket acc. | **−1.3** [−1.6, −1.1] | **+0.2** [+0.04, +0.4] |
| Leak-free images | **−5.5** [−7.6, −3.4] | +1.2 [−0.8, +3.2] |

- With the longer training the attention model overtakes on pixel-level fill (F1 +0.7)
  and on bucket accuracy (+0.2, small but significant), and the leak-free gap disappears
  (+1.2, n.s.). The "it just needs more training" explanation holds for those metrics.
- On **gap closure** removing attention is still better, and the difference grew
  (−1.9 → −3.6 points). So the bottleneck attention does not help close gaps at either
  budget; its benefit is a modest gain in fill precision at the cost of 19M params and
  ~15% inference time.
- Both 9,155-step models are below the released checkpoint on gap closure (59.1 / 62.6
  vs 70.5) despite the same step count, though `full` beats it on fill F1. The released
  weights come from the original notebook run (unseeded, `--seed -1`), so run-to-run
  variance on gap closure may be comparable to the ablation differences (untested); a
  multi-seed run would be needed before claiming a ranking on that metric.
