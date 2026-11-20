# Development results (RQ1)

Dates: logistic regression 2026-11-16, gradient boosting and the text encoder 2026-11-18;
selection applied 2026-11-20. These come from the RQ1 development run: the train and
validation questions of all seven benchmarks (8,307 questions), scored out of fold. They
are development estimates, not final results. The locked test results are in
`docs/RQ1_RESULTS.md` and played no part in anything on this page.

Source files: `results/dev_selection/<model>/`, `results/dev_selection/selection.json`,
`results/dev_selection/diagnostics.json`, `results/dev_selection/dev_tradeoff.png`.
Procedure: `docs/PROTOCOL.md` section 6. Commands: `README.md`, "Running a model".

## Status of the three models

| Model | Owner | Development run |
|---|---|---|
| Regularized logistic regression | Rahul | done, below |
| Gradient boosting | Pravalhika Kurapati | done, below |
| Fine-tuned text encoder (ModernBERT-base) | Priyanka | done, below |

Selected model: **logistic regression**, named on 2026-11-20 from these development results
only (see "Selection" below). Test results played no part in that choice.

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

## Gradient boosting

Settings chosen: max_depth = 3, learning_rate = 0.05, without the simple prompt statistics
(lowest mean calibration MSE across the five rotations). Same TF-IDF features as logistic
regression. The head is a three-class gradient boosting classifier; gain is P(+1) − P(−1).

Cutoff tuned for a target on one fold, then applied to a fold the router never saw:

| Target loss | Gross savings (95% CI) | Actual loss, pp (95% CI) | Sent to Flash |
|---|---:|---:|---:|
| 0 pp | 0.6% (0.4 to 0.8) | 0.07 (−0.05 to 0.19) | 3.1% |
| **1 pp** | **4.7% (4.2 to 5.3)** | **1.12 (0.81 to 1.43)** | 18.2% |
| 2 pp | 12.8% (11.7 to 13.9) | 2.62 (2.16 to 3.08) | 33.8% |
| 3 pp | 17.7% (16.3 to 19.0) | 3.35 (2.81 to 3.90) | 40.2% |
| 5 pp | 27.3% (25.7 to 29.0) | 4.89 (4.21 to 5.53) | 51.7% |

All five targets were declared in advance for this model.

Intervals are paired bootstraps over questions with decisions fixed. They do not include
variability from refitting or from choosing cutoffs.

![Development trade-off](../results/dev_selection/dev_tradeoff.png)

### By benchmark at the 1 pp target

| Benchmark | Questions | Sent to Flash | Gross savings | Accuracy loss, pp |
|---|---:|---:|---:|---:|
| SimpleQA | 3,459 | 1.2% | 1.6% | 0.26 |
| MMLU-Pro | 2,394 | 57.9% | 29.8% | 3.13 |
| HLE | 1,312 | 1.2% | 0.6% | 0.23 |
| LiveCodeBench | 844 | 2.8% | 0.4% | 0.24 |
| GPQA | 158 | 9.5% | 9.3% | 0.63 |
| LiveMathBench | 93 | 31.2% | 17.3% | 2.15 |
| AIME | 47 | 8.5% | 1.8% | 2.13 |

### What this shows

1. **Routing saves a modest share of the bill, less than logistic on the same questions.**
   About 4.7% at about 1 point of accuracy loss (logistic was 6.4% at 1.1 points). That is
   4.2% of the bill per point given up; always using Flash saves 4.6% per point. A router
   that knew the outcome would save 53% and gain 4.9 points (a ceiling for context, not a
   baseline).
2. **The target was roughly met here, but not reliably.** The pooled loss at the 1-point
   target was 1.12, and across the five folds it ranged from 0.73 to 1.44. At the 2-point
   target it was 2.62.
3. **Almost all of the saving comes from MMLU-Pro.** The router sends 58% of MMLU-Pro to
   Flash, about 3% of LiveCodeBench, and about 1% of SimpleQA and HLE. MMLU-Pro alone
   loses 3.1 points; the pooled target does not protect a single benchmark. Unlike logistic,
   it also sends a large share of LiveMathBench to Flash (31%).
4. **Coding is mostly left with GPT-5.** The router sends 2.8% of LiveCodeBench to Flash
   (logistic sent almost none).
5. **The router does pick better individual questions.** Shuffling its choices inside
   each benchmark, keeping the number sent to Flash, gives a 1.33-point loss on average
   (range 1.11 to 1.58) against the router's 1.12. Only 4% of 2,000 shuffles were as
   accurate. The shuffles save more (6.3% against 4.7%).
6. **The signal is weak overall.** Predicted gain correlates 0.15 with the outcome and
   explains 2.2% of its variance.
7. **Provisional selection still prefers logistic.** Both models meet the development loss
   check. With the encoder deferred, GBM saves about 1.7 points less than logistic at 1 pp
   (90% CI about −2.3 to −1.2), so it does not replace logistic under the selection rule.
   The final choice, with the encoder included, is under "Selection" below.

### Retrospective view at equal loss

Read off the pooled out-of-fold predictions afterwards. This describes the curve; it is
not what a tuned cutoff achieved, and it is slightly optimistic.

| Actual loss | Gross savings |
|---|---:|
| 0.5 pp | 2.2% |
| 1.0 pp | 4.3% |
| 1.5 pp | 6.3% |
| 2.0 pp | 8.5% |

### Compute

Forty fits in about two minutes on a laptop CPU. Prediction takes about 0.13 ms per
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

Cutoff tuned for a target on one fold, then applied to a fold the router never saw (from
`results/dev_selection/selection.json`):

| Target loss | Gross savings (95% CI) | Actual loss, pp (95% CI) | Sent to Flash |
|---|---:|---:|---:|
| 0 pp | 0.7% (0.5 to 0.9) | 0.00 (−0.12 to 0.12) | 1.8% |
| **1 pp** | **6.7% (6.1 to 7.4)** | **1.23 (0.86 to 1.61)** | 22.2% |
| 2 pp | 11.4% (10.5 to 12.4) | 2.17 (1.68 to 2.65) | 32.1% |
| 3 pp | 16.5% (15.4 to 17.7) | 3.14 (2.60 to 3.71) | 40.7% |
| 5 pp | 25.7% (24.1 to 27.4) | 4.85 (4.17 to 5.58) | 55.0% |

All five targets were declared in advance for this model. Intervals are paired bootstraps
over questions with decisions fixed. They do not include variability from refitting or
from choosing cutoffs.

### By benchmark at the 1 pp target

| Benchmark | Questions | Sent to Flash | Gross savings | Accuracy loss, pp |
|---|---:|---:|---:|---:|
| SimpleQA | 3,459 | 10.2% | 11.6% | 0.90 |
| MMLU-Pro | 2,394 | 61.1% | 33.7% | 2.76 |
| HLE | 1,312 | 1.2% | 0.5% | 0.23 |
| LiveCodeBench | 844 | 0.0% | 0.0% | 0.00 |
| GPQA | 158 | 8.2% | 4.2% | 1.27 |
| LiveMathBench | 93 | 3.2% | 1.3% | 0.00 |
| AIME | 47 | 0.0% | 0.0% | 0.00 |

### What this shows

1. **About the same as logistic regression at the 1-point target.** 6.7% saved for 1.23
   points lost, against logistic's 6.4% for 1.08. The savings difference is +0.27 points
   (90% CI −0.36 to 0.88), so the two cannot be told apart on these questions.
2. **The saving comes from MMLU-Pro and, unlike the other two models, also SimpleQA.** It
   sends 61% of MMLU-Pro and 10% of SimpleQA to Flash, and keeps coding, AIME and almost all
   of HLE with GPT-5.
3. **It predicts no better than the simpler models.** Out-of-fold advantage error 0.2215,
   against 0.2221 for logistic regression and 0.2228 for gradient boosting.
4. The within-benchmark shuffle, correlation and per-fold figures given for the other two
   models were not computed for the encoder on development questions. Its shuffle check on
   the test is in `docs/RQ1_RESULTS.md`.

## Selection

Applied on 2026-11-20 with `scripts/06_select_algorithm.py` (output
`results/dev_selection/selection.json`), using the rule in `docs/PROTOCOL.md` section 6,
which has not changed since 2026-11-16. Development results only (protocol log entry 19).

| Model | Saved at 1 pp | Lost, pp | Qualifies (≤ 1.5 pp) | Savings minus logistic (90% CI) | Replaces logistic? |
|---|---:|---:|---|---:|---|
| Logistic regression | 6.42% | 1.08 | yes | | |
| Gradient boosting | 4.73% | 1.12 | yes | −1.69 (−2.26 to −1.15) | no |
| Text encoder | 6.69% | 1.23 | yes | +0.27 (−0.36 to 0.88) | no: under the 1.0-point minimum, interval includes zero |

- **Selected: logistic regression** (C = 0.3, with the prompt statistics). It leads the
  report; the other two models are reported alongside it.
- **Highest savings without the parsimony guard: the text encoder** (6.69%). The protocol
  requires it to be reported alongside the selected model.
- The provisional selection with the encoder deferred gave the same answer.
- Each model's test result records a hash of its development summary, taken when the test
  ran. All three match the files used here, so no development result changed after any
  test.

For context only, not part of the rule: at the 3-point target the same rule would also keep
logistic regression (gradient boosting saves 0.24 points more, the encoder 0.95 less).

## How this differs from earlier runs

Earlier development runs used three benchmarks only (SimpleQA, GPQA, filtered MMLU-Pro) and
showed about 14% savings at 1.5 points of loss. Those questions are the ones where routing
is easiest. The lower figure here reflects the full mix, including HLE and coding, where
GPT-5 is much stronger and there is less to save. The three-benchmark run is kept as the
RQ2 pipeline's development run (`results/rq2_dev_selection/`, written up in
`docs/archive/DEV_RESULTS_source_pool.md`).

## Not done

- Development diagnostics (within-benchmark shuffle, correlation, per-fold range) for the
  text encoder.
- The text encoder was trained with one seed; its variation across seeds was not measured.
