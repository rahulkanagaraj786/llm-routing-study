# Development results and algorithm selection

Date: 2026-11-16. Selection on the source pool: the train+validation rows of SimpleQA,
GPQA, and MMLU-Pro without its math and computer-science questions, with one question
removed because it duplicates an HLE item (5,640 questions).

No router from the revised protocol has been evaluated on the locked splits. These are
development estimates, not final results.

Source files: `results/dev_selection/selection.json`, per-candidate
`oof_predictions.csv` and `summary.json`, `diagnostics.json`, `dev_tradeoff.png`.
Procedure: `docs/PROTOCOL.md` section 6.

## Outcome (provisional)

**Among the two CPU candidates, the pre-specified rule picks logistic regression**
(C = 0.3, TF-IDF plus the ten prompt statistics). Both candidates meet the 1.5 pp cap on
out-of-fold loss, and logistic regression saves more.

This is not the final selection. ModernBERT has not been run on this pool; the team is
waiting for one comparable GPU run (1,024 tokens, two learning rates, one seed) and will
then apply the rule to all three. If GPU time falls through, the comparison is reported
as CPU-only. The final evaluation is on hold until then.

## Baselines on the same questions

| Policy | Accuracy | Recorded cost | Savings vs GPT-5 | Accuracy loss |
|---|---:|---:|---:|---:|
| Always GPT-5 | 62.41% | $77.50 | 0% | 0 pp |
| Always Flash | 48.44% | $10.19 | 86.9% | 13.97 pp |

## Candidate comparison (out-of-fold, cutoff chosen on a separate fold)

This is the prospective comparison: the cutoff is tuned for an allowance on one fold and
then applied, frozen, to a fold the router never saw.

| Candidate | Allowance | Gross savings (95% CI) | Actual loss, pp (95% CI) | Sent to Flash |
|---|---|---:|---:|---:|
| Logistic regression | 0 pp | 3.7% (2.9 to 4.7) | 0.23 (0.00 to 0.46) | 5.4% |
| | **1 pp** | **14.1% (12.6 to 15.6)** | **1.49 (1.03 to 1.95)** | 23.9% |
| | 2 pp | 25.4% (23.6 to 27.1) | 2.30 (1.70 to 2.89) | 36.8% |
| Random forest | 0 pp | 1.5% (1.1 to 2.0) | 0.02 (−0.12 to 0.16) | 2.6% |
| | **1 pp** | **10.2% (9.1 to 11.4)** | **1.21 (0.80 to 1.62)** | 17.8% |
| | 2 pp | 17.5% (15.9 to 19.0) | 1.93 (1.40 to 2.46) | 28.1% |

Intervals are paired bootstraps over questions with decisions fixed. They do not include
variability from refitting or from choosing cutoffs.

Selection rule at 1 pp:
1. Eligibility (out-of-fold loss ≤ 1.5 pp): both qualify, logistic regression at 1.489 pp
   and the forest at 1.206 pp.
2. Highest savings among qualified routers: logistic regression.
3. Parsimony guard: logistic regression is also the simpler router, so nothing changes.

Neither interval establishes that the router reliably stays under 1.5 pp: logistic
regression's reaches 1.95 pp and the forest's 1.62 pp.

![Development trade-off](../results/dev_selection/dev_tradeoff.png)

## The selection has flipped on very small changes

| Run | Pool | Logistic regression | Random forest | ModernBERT | Rule's pick |
|---|---|---|---|---|---|
| 1 | Unfiltered, 6,011 | 16.2% at 1.43 pp | 11.5% at 1.26 pp | 9.7% at 0.78 pp | Logistic |
| 2 | Filtered, 5,641 | 16.2% at 1.68 pp | 10.9% at 1.35 pp | not run | Forest (logistic over the cap) |
| 3 | Filtered, one duplicate removed, 5,640 | 14.1% at 1.49 pp | 10.2% at 1.21 pp | not run | Logistic |

Between runs 2 and 3 a single question left the pool. It sat in the fold used to tune one
rotation's cutoff, and that cutoff moved from 0.103 to 0.085. That rotation's result went
from 25.4% savings at 2.46 pp loss to 17.8% at 1.41 pp, which pulled the pooled loss from
1.68 to 1.49 pp, across the 1.5 pp cap. The cutoff search picks the cheapest policy that
just satisfies the constraint, so it is sensitive to individual questions.

The team's decision is to keep the rule as written. Runs 1 and 2 are kept in
`results/dev_selection_v1_unfiltered/` and
`results/dev_selection_v2_with_boundary_duplicate/`.

## Retrospective comparison at equal loss

Computed after the fact by choosing cutoffs on the pooled out-of-fold predictions. It
shows how the two trade-off curves compare; it is not what a tuned cutoff achieved, and it
is optimistic for both routers alike. Keep it separate from the table above.

| Actual loss | Logistic | Forest |
|---|---:|---:|
| 0.5 pp | 6.06% | 6.28% |
| 1.0 pp | 10.82% | 8.65% |
| 1.5 pp | 16.36% | 13.10% |
| 2.0 pp | 22.61% | 17.41% |

## What the comparison shows

1. **The cutoff missed its target.** Tuned for a 1 pp loss, logistic regression lost
   1.49 pp and the forest 1.21 pp on the evaluation folds. Across folds, logistic
   regression ranged from 7.9% to 17.8% savings and 0.98 to 2.13 pp loss; the forest from
   7.5% to 14.6% and 0.98 to 1.58 pp.

2. **Savings are concentrated in MMLU-Pro.** At the 1 pp allowance
   (savings / loss / share sent to Flash):

   | Candidate | MMLU-Pro | SimpleQA | GPQA |
   |---|---|---|---|
   | Logistic | 31.4% / 3.46 pp / 58% | 5.5% / 0.35 pp / 4% | 8.9% / 1.27 pp / 7% |
   | Forest | 20.9% / 2.22 pp / 42% | 5.8% / 0.66 pp / 5% | 0.0% / 0.00 pp / 0% |

   A pooled allowance does not protect any single benchmark: MMLU-Pro alone loses 2 to
   3.5 points.

3. **Within a benchmark the routers do pick better questions, at a cost in savings.**
   Shuffling each router's choices inside every benchmark, keeping the number sent to
   Flash (2,000 shuffles):

   | Candidate | Allowance | Router: savings / loss | Shuffled: savings / loss (95% range of loss) | Accuracy advantage | Shuffles at least as accurate |
   |---|---|---|---|---:|---:|
   | Logistic | 0 pp | 3.7% / 0.23 pp | 3.9% / 0.52 pp (0.28 to 0.76) | +0.29 pp | 0.7% |
   | | 1 pp | 14.1% / 1.49 pp | 16.1% / 1.96 pp (1.60 to 2.32) | +0.47 pp | 0.6% |
   | | 2 pp | 25.4% / 2.30 pp | 26.2% / 3.32 pp (2.89 to 3.72) | +1.02 pp | 0.0% |
   | Forest | 0 pp | 1.5% / 0.02 pp | 1.8% / 0.22 pp (0.05 to 0.39) | +0.20 pp | 0.9% |
   | | 1 pp | 10.2% / 1.21 pp | 12.1% / 1.53 pp (1.15 to 1.90) | +0.33 pp | 5.2% |
   | | 2 pp | 17.5% / 1.93 pp | 19.5% / 2.52 pp (2.11 to 2.94) | +0.59 pp | 0.3% |

   The routers are more accurate than policies with the same per-benchmark routing
   rates, so they are not only recognising the benchmark. But they save about 2 points
   less than those policies at the 1 pp setting: the questions they send to Flash are the
   ones where sending is safer, and those tend to be the cheaper ones for GPT-5 too.

4. **The question-level signal is weak overall.** Predicted gain correlates 0.12–0.14
   with Y and explains 1–2% of its variance. A router that knew Y would save 70.5% with a
   5.8-point accuracy gain on these questions (a ceiling for context, not a baseline).

## Search and compute record

| Candidate | Chosen configuration | Fits | Total fit time | Prediction time | Device |
|---|---|---:|---:|---:|---|
| Logistic | C = 0.3, with statistics | 50 | 56 s | 0.09 ms/question | CPU (M2 Pro) |
| Forest | 300 trees, leaf 20, max_features 0.1, with statistics | 25 | 72 s | 0.13 ms/question | CPU, 8 threads |
| ModernBERT | not run on this pool | | | | waiting for a GPU |

## Gross versus net savings

All savings above are gross: recorded API cost only. Both CPU routers predict in about
0.1 ms per question. At an assumed $0.10 per CPU-hour that is under $0.00000001 per
question, against an average GPT-5 call of $0.0137 in this pool, so net and gross
savings agree to two decimal places. An always-on server, idle capacity, added latency
and engineering time were not measured.

## ModernBERT

- First run (unfiltered pool, 512-token prompts on this Mac's MPS backend): 9.7% savings
  at 0.78 pp loss; at equal loss it was close to logistic regression and never ahead.
  Kept as pilot evidence (`docs/archive/DEV_RESULTS_v1_unfiltered.md`).
- Source pool: one GPU run with a fixed budget (1,024 tokens, learning rates 2e-5 and
  5e-5, one seed, same folds): `python scripts/05_dev_compare.py modernbert`, then
  `python scripts/06_select_algorithm.py`.
- If no GPU run happens: `python scripts/06_select_algorithm.py --drop modernbert`, and
  the comparison is reported as CPU-only.

## Not done

- No router from the revised protocol has been evaluated on the locked test split or any
  held-out benchmark. `scripts/07_final_eval.py --unlock-final` runs that once and is
  blocked while a candidate is deferred. It was rehearsed with `--dry-run` on development
  rows only; `scripts/09_report_final.py` turns its output into tables and figures.
- No GPU run.
