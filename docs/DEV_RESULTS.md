# Development results (RQ1)

Date: 2026-11-16. These come from the RQ1 development run: the train and validation
questions of all seven benchmarks (8,307 questions), scored out of fold. No router from
the revised protocol has been evaluated on the locked test split. These are development
estimates, not final results.

Source files: `results/dev_selection/<model>/`, `results/dev_selection/selection.json`,
`results/dev_selection/diagnostics.json`, `results/dev_selection/dev_tradeoff.png`.
Procedure: `docs/PROTOCOL.md` section 6. How to produce these for another model:
`docs/TEAM_GUIDE.md`.

## Status of the three models

| Model | Owner | Development run |
|---|---|---|
| Regularized logistic regression | Rahul | done, below |
| Gradient boosting | to be assigned | not yet written |
| Fine-tuned text encoder | to be assigned | not yet written |

The final model is named from this table once all three have development results. Test
results play no part in that choice.

## Baselines on the same questions

| Policy | Accuracy | Recorded cost | Savings vs GPT-5 | Accuracy loss |
|---|---:|---:|---:|---:|
| Always GPT-5 | 61.07% | $238.46 | 0% | 0 pp |
| Always Flash | 45.62% | $69.66 | 70.8% | 15.44 pp |

## Logistic regression

Settings chosen: penalty C = 0.3, with the ten simple prompt statistics (chosen in four of
the five rotations).

Cutoff tuned for a target on one fold, then applied to a fold the router never saw:

| Target loss | Gross savings (95% CI) | Actual loss, pp (95% CI) | Sent to Flash |
|---|---:|---:|---:|
| 0 pp | 1.0% (0.6 to 1.3) | 0.08 (−0.02 to 0.19) | 2.1% |
| **1 pp** | **6.4% (5.7 to 7.2)** | **1.08 (0.74 to 1.41)** | 20.5% |
| 2 pp | 12.8% (11.8 to 13.9) | 2.36 (1.89 to 2.84) | 33.3% |

Intervals are paired bootstraps over questions with decisions fixed. They do not include
variability from refitting or from choosing cutoffs.

![Development trade-off](../results/dev_selection/dev_tradeoff.png)

### By benchmark at the 1 pp target

| Benchmark | Questions | Sent to Flash | Gross savings | Accuracy loss, pp |
|---|---:|---:|---:|---:|
| SimpleQA | 3,459 | 4.6% | 6.0% | 0.29 |
| MMLU-Pro | 2,394 | 60.8% | 31.9% | 2.92 |
| HLE | 1,312 | 4.1% | 2.2% | 0.53 |
| LiveCodeBench | 844 | 0.1% | 0.0% | 0.00 |
| GPQA | 158 | 14.6% | 11.5% | 0.63 |
| LiveMathBench | 93 | 10.8% | 6.0% | 1.08 |
| AIME | 47 | 4.3% | 0.2% | 2.13 |

### What this shows

1. **Routing saves a modest share of the bill.** About 6% at about 1 point of accuracy
   loss. A router with no skill, sliding along the straight line between always GPT-5 and
   always Flash, would save about 4.6% at 1 point. A router that knew the outcome would
   save 53% and gain 4.9 points (a ceiling for context, not a baseline).
2. **The target was roughly met here, but not reliably.** The pooled loss at the 1-point
   target was 1.08, and across the five folds it ranged from 0.48 to 1.93. At the 2-point
   target it was 2.36.
3. **Almost all of the saving comes from MMLU-Pro.** The router sends 61% of MMLU-Pro to
   Flash, almost none of LiveCodeBench, and about 4% of SimpleQA and HLE. MMLU-Pro alone
   loses 2.9 points; the pooled target does not protect a single benchmark.
4. **Coding is left with GPT-5.** The router sends 0.1% of LiveCodeBench to Flash, which
   is consistent with GPT-5's 26-point lead there.
5. **The router does pick better individual questions.** Shuffling its choices inside
   each benchmark, keeping the number sent to Flash, gives a 1.65-point loss on average
   (range 1.38 to 1.94) against the router's 1.08. No shuffle out of 2,000 was as
   accurate. The shuffles save slightly more (7.3% against 6.4%).
6. **The signal is weak overall.** Predicted gain correlates 0.16 with the outcome and
   explains 2.4% of its variance.

### Retrospective view at equal loss

Read off the pooled out-of-fold predictions afterwards. This describes the curve; it is
not what a tuned cutoff achieved, and it is slightly optimistic.

| Actual loss | Gross savings |
|---|---:|
| 0.5 pp | 3.0% |
| 1.0 pp | 6.2% |
| 1.5 pp | 9.1% |
| 2.0 pp | 11.7% |

### Compute

Fifty fits in about two minutes on a laptop CPU. Prediction takes about 0.17 ms per
question, so the router's own cost is negligible next to an average GPT-5 call of about
$0.029 on these questions. Savings above are gross (API cost only).

## How this differs from earlier runs

Earlier development runs used three benchmarks only (SimpleQA, GPQA, filtered MMLU-Pro) and
showed about 14% savings at 1.5 points of loss. Those questions are the ones where routing
is easiest. The lower figure here reflects the full mix, including HLE and coding, where
GPT-5 is much stronger and there is less to save. The three-benchmark run is kept as the
RQ2 pipeline's development run (`results/rq2_dev_selection/`, written up in
`docs/archive/DEV_RESULTS_source_pool.md`).

## Not done

- No model has been evaluated on the locked test split.
- Gradient boosting and the text encoder have no development results yet.
