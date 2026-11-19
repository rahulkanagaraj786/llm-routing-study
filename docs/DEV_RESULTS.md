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
| Fine-tuned text encoder (ModernBERT-base) | Priyanka | done, below |

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
| 3 pp (added later) | 17.5% (16.2 to 18.8) | 3.35 (2.80 to 3.91) | 40.7% |
| 5 pp (added later) | 26.6% (25.0 to 28.3) | 5.47 (4.79 to 6.19) | 54.0% |

The 3 and 5 point rows were added after logistic regression's first test (protocol log
entry 12). For the other two models all five are declared in advance.

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
   loss, which is 5.9% of the bill per point given up; always using Flash saves 4.6% per
   point. A router that knew the outcome would save 53% and gain 4.9 points (a ceiling for
   context, not a baseline).
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

## Fine-tuned text encoder (ModernBERT-base)

Date: 2026-11-18. ModernBERT-base with a three-class head, fine-tuned end to end
(`src/router/encoder.py`): prompts up to 1,024 tokens, batches of 16 grouped by length,
up to 3 epochs with the calibration fold scored four times per epoch and the best snapshot
kept, fp16 mixed precision, seed 221. Run on a Google Colab Tesla T4 GPU.

Settings chosen: learning rate 2e-5. Both learning rates were fitted in rotation 0 and the
better one was reused in rotations 1 to 4 (six fine-tunes in total).

| Rotation | Learning rates tried | Calibration error (chosen) | Best point (epoch) | Fit time |
|---|---|---:|---:|---:|
| 0 | 2e-5 (0.2295), 5e-5 (0.2305) | 0.2295 | 1.5 | 5.9 min each |
| 1 | 2e-5 | 0.2165 | 2.0 | 7.3 min |
| 2 | 2e-5 | 0.2257 | 0.5 | 3.3 min |
| 3 | 2e-5 | 0.2226 | 0.5 | 3.4 min |
| 4 | 2e-5 | 0.2134 | 1.25 | 5.3 min |
| **Mean** | | **0.2215** | | |

The mean calibration error is the same as logistic regression's (0.2215 against 0.2215 to
four decimals). About 1% of training prompts were longer than 1,024 tokens and were
truncated. Prediction takes about 6.6 ms per question on the T4.

The out-of-fold savings and losses at each target, and the comparison with logistic
regression under the selection rule, come from `scripts/06_select_algorithm.py`, which has
not been run with the encoder yet.

## How this differs from earlier runs

Earlier development runs used three benchmarks only (SimpleQA, GPQA, filtered MMLU-Pro) and
showed about 14% savings at 1.5 points of loss. Those questions are the ones where routing
is easiest. The lower figure here reflects the full mix, including HLE and coding, where
GPT-5 is much stronger and there is less to save. The three-benchmark run is kept as the
RQ2 pipeline's development run (`results/rq2_dev_selection/`, written up in
`docs/archive/DEV_RESULTS_source_pool.md`).

## Not done

- Logistic regression and the text encoder have since been evaluated on the locked test
  split (`docs/RQ1_RESULTS.md`).
- Gradient boosting has no development results yet.
- The selection rule has not been applied to the text encoder's results.
