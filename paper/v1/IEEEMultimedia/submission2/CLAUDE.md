# IEEE MultiMedia revision (submission2) — working notes

## Situation
- Manuscript MM-2026-10-0109, "InkCloser: Line-Art Completion for Paint-Bucket
  Colorization", submitted to IEEE MultiMedia (magazine, `IEEEcsmag` class).
- `../submission1/` = the version that was submitted (frozen, do not edit).
  Reviews: `../submission1/review1_comments.txt` (R1 reject, R2/R3 major revision;
  editor: not recommended in current form, resubmission allowed).
- `./` (submission2) = the revised version being prepared, started as a copy of
  submission1.
- Base (long, journal-style) paper: `../../paper.tex` (paper/v1). New results are
  developed there first and then ported here in magazine style (plain language,
  no equations/hyperparameter symbols, `\textbf{...}` paragraph heads,
  `table*` + `\tablefont` + `tabular*`, uppercase unnumbered sections).

## Changes done so far
1. **Ablation study** (from the base paper's "Ablation Study" section, commits
   0e69155..749f6a2): new section `ABLATION STUDY` between RESULTS and LESSONS AND
   LIMITATIONS, and Table 2 (`tab:ablation`). The base paper's two tables
   (`tab:ablation`, `tab:ablation_long`) were merged into one table with two
   groups (one epoch / training length of the reported model) to save floats.
   Numbers are identical to the base paper. Results data:
   `../../results_ablations/`, `../../results_ablations_9155/`,
   `code/ablations/README.md`.

2. **Softened attention claims** to agree with the ablation: in "Why redrawing
   everything is not enough" the gain is no longer attributed to the attention
   modules (points to the ablation instead), and "Long gaps" now says long-gap
   closure does not depend on the bottleneck attention. The method description
   ("Seeing the whole drawing") still motivates both context modules as design
   rationale. The author decided NOT to remove the bottleneck attention or
   retrain (the +3.6 gap-closure difference is within unmeasured run-to-run
   variance; fillability metrics are tied). The same softening was applied to
   the base paper (`../../paper.tex`), which also softens the intro sentence on
   "global attention" bridging long gaps.

3. **R2 minor comments m1-m7** (text only, also in the base paper where
   relevant): unrounded-values note (m1), definition of the 94.0% region
   recovery (m2), output head channels in the Fig. 2 caption and softened
   skip-connection claim (m3), precision and timing conditions in "Speed" (m4),
   gap length as a proxy in the Fig. 4 caption (m5), regions = 4-connected
   background areas = bucket fills (m6, no new figure), abstract/conclusion
   claims restricted to bucket accuracy (m7). Word count after this: ~4,900
   words of text (texcount) + 6 floats, i.e. ~6,100 of 6,500.

4. **Metric definitions (R1.6, R2.2, R2.3)**, text only, both papers: pixel
   gap closure is not topological (paint-bucket metrics are primary), precise
   region/matching/leak rules (exterior included, <20 px ignored, >=50% of each
   of two regions, k-1 leaks, pooled), leak-free no longer equated with "no
   correction needed". No split/lost-region stats, no topological gap metric.
   Word count now ~5,000 (texcount total) + 6 floats ≈ 6,200 of 6,500.

5. **R3.5, R3.10-R3.12**: one sentence in the baseline paragraph (both papers)
   on why mask-guided/diffusion and blind inpainting models are not baselines;
   R3.10-R3.12 answered in the letter only (no paper change).

6. **Loss and training (R2.7, R1.8, R3.4, R3.8)**: submission2 now defines the
   hole mask, says Dice uses ink probability, describes the local term as
   "within 7 pixels (r=7 in Fig. 2)", gives loss weights 1/1/0.5 (preliminary
   experiments), LRs, weight decay, cosine, clipping, and "model after the last
   training step, without checkpoint selection" (confirmed by the author). Base
   paper: per-image Dice with eps=1, weights not claimed optimal, no checkpoint
   selection. Full formulas only in the letter. The author has no loss curves;
   the letter makes no convergence/plateau claim and gives no epoch count.

7. **Contributions (R1.1, R2.1)**: minimal, by the author's choice: the first
   contribution now says the network preserves intact strokes, unlike Sasaki et
   al., which regenerates the whole drawing (both papers). Dry, short answers.

Review philosophy: see `../../review_strategy.md`.

8. **All remaining comments** (R1.3-R1.5, R1.7, R1.9, R1.10, R2.5, R2.6, R3.1,
   R3.3, R3.6, R3.7, R3.9): letter-only answers, plus small clarifications in
   both papers: gap lengths uniform 4-60 px, cuts remove the full stroke width
   and widths are kept, generation code with parameters and seeds released
   (R1.7, R3.7), "1,000 resamples" for the bootstrap (R1.9). The two references
   suggested by R2 (allen2024leak, asrin2025gap) were added to the related work of
   both papers; in submission2, simoserra2016simplify and casey2021animation were
   removed to stay at 20 references.
   The letter says no near-duplicate check was done (R1.7); the author approved.

All comments now have an answer in review1_answers.txt. Before submitting:
review the letter as a whole, and rebuild the zip if anything changes.

## Files
- `paper.tex` — clean revised manuscript (the one to upload; the editor requires
  NO colored text in the main file).
- `paper_changes.tex/.pdf` — GENERATED annotated version (latexdiff vs
  submission1; additions in blue, deletions hidden). Upload it under "Summary of
  Changes". Never edit by hand; regenerate with `./build.sh`.
- `latexdiff_preamble.tex` — the blue/no-deletion latexdiff style.
- `build.sh` — builds `paper.pdf` and `paper_changes.pdf`.
- `review1_answers.txt` — the actual response letter. Same structure as the
  candidate file; only the ablation-related answers are filled (General item 1,
  R1.2, R2.4, R3.2, R3.13). Fill the other entries as changes are made.
- `review1_answers_CANDIDATE.txt` — the author's brainstorm of possible answers
  and experiments (EXP-1..11). Ideas only: do NOT implement them unless asked.
- `IEEEMultimedia_submission.zip` — source bundle of the clean revision (tex, bbl,
  bib, cls/bst/sty, paper.pdf, figures). Rebuilt 2026-11-19; rebuild again if
  anything changes.

## Constraints (from the editor letter)
- ≤ 6,500 words, figures+tables count 200 words each, ≤ 12 floats, ≤ 20 references
  (there are already 20).
- Current: ~5,200 words (texcount total, incl. captions) + 6 floats × 200 ≈
  6,400. Only ~100 words of headroom left. References: exactly 20.

## Open issues to keep in mind
- The paper says the model was trained for five epochs, but the released model
  was trained for 9,155 steps (~4 epochs of a 30-epoch cosine schedule; see the
  candidate file's "THINGS TO CHECK" and `code/ablations/README.md`). The
  ablation text says "training length of the reported model", "one epoch
  instead of five" and "trained five times longer" (really ~4x), mirroring the
  base paper.
- Abstract says 54,000 training pairs, dataset section says 55,000 pairs in total.

## Workflow rule
- Always rebuild the PDFs after any change: `./build.sh` here (paper.pdf +
  paper_changes.pdf), `latexmk -pdf paper.tex` for the base paper in `../../`.
