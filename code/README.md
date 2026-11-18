# InkCloser code (scripts)

Plain-Python version of `lineartanimdata3_closer_v12.ipynb`, so the paper's experiments
can be run from the command line. The model, loss, metrics and evaluation code are
ported unchanged from the notebook (same `state_dict` keys, so notebook checkpoints and
the released weights load directly).

```
code/
  inkcloser/            library (data, model, loss, metrics, viz, paper_eval)
  download_dataset.py   CODE BLOCK 1   download + unzip the dataset to /content/split
  download_checkpoint.py CODE BLOCK 13b download the released weights
  check_data.py         CODE BLOCK 5   data-loader sanity checks (figures to disk)
  train.py              CODE BLOCK 13  training (+ save / resume)
  evaluate.py           CODE BLOCK 14  paper tables, bootstrap tests, figures
  intro_figures.py      CODE BLOCK 15  one-example PDFs for the intro figure
```

Run everything from the repo root.

## Reproducing the paper

```bash
python code/download_dataset.py                 # -> /content/split
python code/download_checkpoint.py              # -> /content/checkpoints/inkcloser_paper.pt
python code/check_data.py

# Table 1/2/3 numbers for Input, Closing and Ours (~25 min on a T4)
python code/evaluate.py --checkpoint /content/checkpoints/inkcloser_paper.pt \
    --results-dir /content/paper_results

# Merge the Sasaki et al. results (computed outside this repo) into the same tables
python code/evaluate.py --checkpoint /content/checkpoints/inkcloser_paper.pt \
    --extra-results paper/v1/results_sasaki \
    --rename "Sasaki17 pretrained (scale 1)=Sasaki et al. (pretrained)" \
    --rename "Sasaki17 retrained (scratch, mse, adadelta)=Sasaki et al. (retrained)"

# Or only re-aggregate saved per-image CSVs (no GPU, seconds)
python code/evaluate.py --from-csv paper/v1/results paper/v1/results_sasaki --rename ... 
```

Training from scratch (defaults = paper settings: cosine schedule over 30 epochs,
stopped after 5; ~1 h/epoch on a T4):

```bash
python code/train.py --out-dir /content/checkpoints/inkcloser
python code/train.py --out-dir /content/checkpoints/inkcloser --resume /content/checkpoints/inkcloser/last.pt
python code/evaluate.py --checkpoint /content/checkpoints/inkcloser/last.pt
```

The released checkpoint was trained for 9,155 steps (4 epochs + 155 steps of the 5th);
to retrain exactly that length use `--max-steps 9155`.

## Quick check runs

```bash
python code/train.py --out-dir /tmp/smoke --max-steps 20 --eval-batches 2 --preview-every 10
python code/train.py --out-dir /tmp/smoke2 --train-subset 16 --stop-after 2 --eval-batches 1
python code/evaluate.py --checkpoint ... --max-images 40 --results-dir /tmp/eval40
```

## Verification done

- `--from-csv paper/v1/results paper/v1/results_sasaki` reproduces Tables 1 and 2 of the
  paper exactly (point values, CIs and paired differences).
- With the released weights on the first 40 test images, the per-image records for
  Input / Closing are identical to `paper/v1/results/per_image_metrics.csv`; for Ours all
  summed counts match, with ±1 px differences on a few images (fp16 nondeterminism).
- Training: loss decreases, checkpoints save/resume (mid-epoch and at epoch boundaries),
  cosine LR matches the notebook.

Not included: the Sasaki et al. conversion / retraining code (not in this repo); its
saved results are merged via `--extra-results`. Training is seeded (`--seed 0`); the
notebook was not, so a retrained model will differ slightly from the released weights.
