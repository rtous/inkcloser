# Review strategy

How the author wants reviewer comments to be addressed (applies to every
revision, e.g. `IEEEMultimedia/submission2`).

1. **Avoid new experiments and drastic changes.** Prefer, in this order:
   - adding a clarification or precise definition in the text;
   - stating that the request is not feasible here, or is future work (and saying
     so in the limitations when appropriate);
   - answering with the reason why the request does not apply or is not needed.
     Never state the disagreement explicitly ("we respectfully disagree",
     "we disagree"): just give the answer and its justification.
2. **Experiments only when critical**, i.e. when a comment cannot be answered
   credibly without data (as with the ablation study requested by all three
   reviewers). Ask the author before running any.
3. **Go one comment (or group of related comments) at a time**, easiest first.
   Propose how to address it, wait for approval, then implement.
4. **Be honest.** Report results as they are (e.g. the bottleneck attention does
   not help gap closure) and soften claims rather than overstating them. Never
   claim in the response letter a change that was not made.
5. **Keep the base paper in sync.** Apply each change to the base paper
   (`paper/v1/paper.tex`) too, when it makes sense.
6. **Respect the venue limits** (IEEE MultiMedia: 6,500 words, floats count 200
   words each, 20 references). Clarifications must be short.
7. **Response letter** (`review1_answers.txt`): concise, same structure as
   `review1_answers_CANDIDATE.txt`; the candidate file holds ideas only.
8. **Always rebuild the PDFs** after any change (clean and highlighted versions).
