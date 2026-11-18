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
python code/ablations/run_ablations.py            # train + evaluate + report (~6-7 h on a T4)
python code/ablations/run_ablations.py --stages report
python code/ablations/run_ablations.py --check --ckpt-root /tmp/abl --results-dir /tmp/abl_res
```

Checkpoints and per-variant per-image metrics go to `/content/checkpoints/ablations/<key>/`;
tables go to `paper/v1/results_ablations/` (`table_ablation.tex`, `ablation_table.csv`,
and the standard paper tables / paired bootstrap against the full model). The script
skips finished stages, so rerun it after an interruption.
