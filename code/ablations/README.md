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
