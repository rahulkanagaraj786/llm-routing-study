# RQ1 results

RQ1: how much can a router save compared with always using GPT-5, and what does that do
to accuracy?

Each model is scored once on the locked test split (2,060 questions from all seven
benchmarks), after its settings were fixed in its development run and its cutoffs were
tuned on the validation split and frozen. Full tables and figures per model are in
`results/final/<model>/report/`.

| Model | Owner | RQ1 test |
|---|---|---|
| Regularized logistic regression | Rahul | run on 2026-11-16, below |
| Gradient boosting | to be assigned | pending |
| Fine-tuned text encoder (ModernBERT-base) | Priyanka | run on 2026-11-18, below |

A side-by-side comparison of the two tested models is at the end of this page.

The final model is named from the development results of all three
(`docs/DEV_RESULTS.md`), not from the numbers on this page.

## Logistic regression

Trained on 6,216 questions, cutoffs tuned on 2,091 validation questions. Settings from the
development run: C = 0.3 with the simple prompt statistics.

### Headline

| Policy | Accuracy | Recorded cost | Bill saved (95% CI) | Accuracy lost, pp (95% CI) | Sent to Flash |
|---|---:|---:|---:|---:|---:|
| Always GPT-5 | 61.84% | $61.30 | 0% | 0 | 0% |
| Router, tuned for 0 pp | 61.80% | $61.23 | 0.1% (0.0 to 0.2) | 0.05 (0.00 to 0.15) | 0.4% |
| **Router, tuned for 1 pp** | **60.68%** | **$57.79** | **5.7% (4.7 to 6.9)** | **1.17 (0.49 to 1.84)** | **19.1%** |
| Router, tuned for 2 pp | 59.85% | $54.96 | 10.3% (8.8 to 12.1) | 1.99 (1.12 to 2.85) | 30.3% |
| Router, tuned for 3 pp (added later) | 58.64% | $51.42 | 16.1% (13.9 to 18.5) | 3.20 (2.15 to 4.26) | 40.5% |
| Router, tuned for 5 pp (added later) | 56.75% | $45.46 | 25.8% (22.8 to 29.2) | 5.10 (3.77 to 6.45) | 53.8% |
| Always Flash | 45.87% | $14.85 | 75.8% | 15.97 | 100% |

The 0, 1 and 2 point limits were fixed before any test. The 3 and 5 point limits were added
after the first test result was known, to show how the trade-off continues for a company
willing to give up more accuracy (protocol log entry 12). They were tuned on validation and
scored exactly like the others, and adding them left the 0, 1 and 2 results unchanged.

![Trade-off on the locked test](../results/final/logistic/report/fig1_rq1_tradeoff.png)

### Did the accuracy target hold?

| Tuned for | Loss on validation | Loss on test |
|---|---:|---:|
| 0 pp | 0.00 | 0.05 |
| 1 pp | 0.96 | 1.17 |
| 2 pp | 1.96 | 1.99 |
| 3 pp (added later) | 2.97 | 3.20 |
| 5 pp (added later) | 4.97 | 5.10 |

The targets were close but not guaranteed. At the 1-point target the test loss was 1.17,
and its interval (0.49 to 1.84) includes values well above 1. In the bootstrap, the loss
stayed within 1 point in about a third of resamples. At the 2-point target the test loss
landed almost exactly on the target.

![Target versus actual](../results/final/logistic/report/fig3_allowance_check.png)

### Where the saving comes from (1 pp target)

| Benchmark | Questions | GPT-5 right | Flash right | Sent to Flash | Bill saved | Accuracy lost, pp |
|---|---:|---:|---:|---:|---:|---:|
| SimpleQA | 856 | 50.0% | 31.1% | 4.7% | 6.1% | −0.12 |
| MMLU-Pro | 599 | 87.6% | 81.3% | 55.3% | 30.2% | 3.84 |
| HLE | 319 | 23.8% | 6.3% | 6.9% | 2.6% | 0.63 |
| LiveCodeBench | 211 | 85.8% | 58.3% | 0.0% | 0.0% | 0.00 |
| GPQA | 40 | 87.5% | 65.0% | 2.5% | 2.1% | 0.00 |
| LiveMathBench | 24 | 83.3% | 66.7% | 0.0% | 0.0% | 0.00 |
| AIME | 11 | 81.8% | 63.6% | 0.0% | 0.0% | 0.00 |
| Equal-benchmark average | | | | | 5.9% | 0.62 |

![By benchmark](../results/final/logistic/report/fig2_rq1_by_benchmark.png)

The router sends more than half of MMLU-Pro to Flash and almost nothing from coding or
math. Nearly all of the pooled accuracy loss falls on MMLU-Pro, which loses 3.8 points.

### How much does each point of accuracy buy?

| Policy | Bill saved per point of accuracy lost |
|---|---:|
| Router, tuned for 1 pp | 4.9% |
| Router, tuned for 2 pp | 5.2% |
| Router, tuned for 3 pp | 5.0% |
| Router, tuned for 5 pp | 5.1% |
| Always Flash | 4.7% |

At the five tuned limits (0 to 5 points) the router buys slightly more saving per point of
accuracy than always using Flash does, about 5% of the bill per point against 4.7%. Read
off the full test curve, it sits up to about 2 points of savings above the straight line
between the two baselines at small losses, crosses it near 6 points of loss, and falls
below it at large losses (about 1.7 points under at 10 points of loss, about 5 under at
12). So the router is only useful for small accuracy losses; past roughly 6 points, simply
mixing GPT-5 and Flash in fixed proportions would save more. Its value on this test is that
it lets you choose a small accuracy loss, not that it makes each point of accuracy much
cheaper to give up.

### Is the router more than benchmark recognition?

Shuffling the router's choices inside each benchmark, keeping the number sent to Flash
(2,000 shuffles):

| Tuned for | Router: saved / lost | Shuffled: saved / lost (95% range of loss) | Router's accuracy advantage | Shuffles at least as accurate |
|---|---|---|---:|---:|
| 1 pp | 5.7% / 1.17 pp | 7.5% / 1.58 pp (1.07 to 2.14) | +0.42 pp | 7.6% |
| 2 pp | 10.3% / 1.99 pp | 13.2% / 2.80 pp (2.18 to 3.45) | +0.81 pp | 0.5% |
| 3 pp | 16.1% / 3.20 pp | 20.3% / 4.22 pp (3.50 to 4.95) | +1.01 pp | 0.4% |
| 5 pp | 25.8% / 5.10 pp | 30.2% / 6.53 pp (5.68 to 7.38) | +1.43 pp | 0.1% |

The router is more accurate than policies with the same per-benchmark routing rates, and
the advantage grows as more questions are sent to Flash: only suggestive at 1 point,
clear from 2 points on. It also saves
less than they do, because the questions it chooses to send to Flash are cheaper ones.

### If the workload had more coding

Same frozen decisions, questions reweighted so that LiveCodeBench makes up a larger share
(it is 10.2% of the test split):

| Coding share | Saved / lost at 1 pp target | Saved / lost at 5 pp target |
|---|---:|---:|
| 10% | 5.7% / 1.17 | 26.0% / 5.11 |
| 25% | 4.2% / 0.97 | 19.2% / 4.34 |
| 50% | 2.3% / 0.65 | 10.9% / 3.05 |

The router keeps coding questions on GPT-5, so the more coding there is, the less it saves.

### Sensitivity checks

| Check | Bill saved (1 pp target) | Accuracy lost, pp |
|---|---:|---:|
| Main result | 5.7% | 1.17 |
| HLE with repaired costs (2,167 questions) | 5.0% | 1.11 |
| Costs at list prices | 5.7% | 1.17 |

### Router overhead

Fitting took 3.6 seconds and prediction about 0.19 ms per question on a laptop CPU. That is
negligible next to an average GPT-5 call of about $0.03, so gross and net savings are
effectively the same. Serving costs (an always-on machine, added delay) were not measured.

### Summary for logistic regression

- Tuned for a 1-point loss, the router cut the bill by 5.7% and lost 1.17 points.
- Tuned for 2 points, it cut the bill by 10.3% and lost 1.99 points.
- With looser limits added afterwards: 16.1% saved for 3.20 points, 25.8% for 5.10 points.
- The saving comes almost entirely from sending MMLU-Pro questions to Flash.
- Per point of accuracy given up, it saves slightly more than always using Flash up to about
  6 points of loss; beyond that, less.
- Development and test agree: the development run predicted 6.4% at 1.08 points.

## Fine-tuned text encoder (ModernBERT-base)

ModernBERT-base with a three-class head, fine-tuned end to end (`src/router/encoder.py`).
Trained on 6,216 questions, cutoffs tuned on 2,091 validation questions. Settings from the
development run: learning rate 2e-5 (chosen over 5e-5). Prompts up to 1,024 tokens, batches
of 16, up to 3 epochs with early stopping on the validation questions, fp16 mixed precision,
seed 221. Trained on a Google Colab Tesla T4 GPU.

All five accuracy limits (0, 1, 2, 3 and 5 points) were set before this model's test.

### Headline

| Policy | Accuracy | Recorded cost | Bill saved (95% CI) | Accuracy lost, pp (95% CI) | Sent to Flash |
|---|---:|---:|---:|---:|---:|
| Always GPT-5 | 61.84% | $61.30 | 0% | 0 | 0% |
| Router, tuned for 0 pp | 61.55% | $59.72 | 2.6% (2.0 to 3.3) | 0.29 (−0.2 to 0.7) | 7.6% |
| **Router, tuned for 1 pp** | **60.58%** | **$58.28** | **4.9% (3.8 to 6.1)** | **1.26 (0.6 to 2.0)** | **17.6%** |
| Router, tuned for 2 pp | 58.88% | $53.39 | 12.9% (10.9 to 15.0) | 2.96 (1.9 to 4.1) | 36.2% |
| Router, tuned for 3 pp | 57.82% | $49.00 | 20.1% (17.5 to 22.7) | 4.03 (2.7 to 5.3) | 48.1% |
| Router, tuned for 5 pp | 55.53% | $44.13 | 28.0% (24.8 to 31.4) | 6.31 (4.8 to 7.8) | 59.4% |
| Always Flash | 45.87% | $14.85 | 75.8% | 15.97 | 100% |

![Trade-off on the locked test](../results/final/encoder/report/fig1_rq1_tradeoff.png)

### Did the accuracy target hold?

| Tuned for | Loss on validation | Loss on test |
|---|---:|---:|
| 0 pp | 0.00 | 0.29 |
| 1 pp | 0.96 | 1.26 |
| 2 pp | 1.96 | 2.96 |
| 3 pp | 2.97 | 4.03 |
| 5 pp | 4.97 | 6.31 |

The cutoffs hit their targets on validation but overshot on the test at every limit, by
about 0.3 points at 0 and 1 point and by about 1 to 1.3 points at 2, 3 and 5 points.

![Target versus actual](../results/final/encoder/report/fig3_allowance_check.png)

### Where the saving comes from (1 pp target)

| Benchmark | Questions | GPT-5 right | Flash right | Sent to Flash | Bill saved | Accuracy lost, pp |
|---|---:|---:|---:|---:|---:|---:|
| SimpleQA | 856 | 50.0% | 31.1% | 14.5% | 18.0% | 0.93 |
| MMLU-Pro | 599 | 87.6% | 81.3% | 39.4% | 16.8% | 2.84 |
| HLE | 319 | 23.8% | 6.3% | 0.0% | 0.0% | 0.00 |
| LiveCodeBench | 211 | 85.8% | 58.3% | 0.0% | 0.0% | 0.00 |
| GPQA | 40 | 87.5% | 65.0% | 2.5% | 3.8% | 0.00 |
| LiveMathBench | 24 | 83.3% | 66.7% | 4.2% | −12.2% | 4.17 |
| AIME | 11 | 81.8% | 63.6% | 0.0% | 0.0% | 0.00 |
| Equal-benchmark average | | | | | 3.8% | 1.13 |

![By benchmark](../results/final/encoder/report/fig2_rq1_by_benchmark.png)

Unlike logistic regression, the encoder splits its savings between SimpleQA and MMLU-Pro.
It keeps all of HLE, coding and AIME on GPT-5. (The negative saving on LiveMathBench is one
question, which happened to cost more on Flash than on GPT-5.)

### How much does each point of accuracy buy?

| Policy | Bill saved per point of accuracy lost |
|---|---:|
| Router, tuned for 1 pp | 3.9% |
| Router, tuned for 2 pp | 4.4% |
| Router, tuned for 3 pp | 5.0% |
| Router, tuned for 5 pp | 4.4% |
| Always Flash | 4.7% |

Per point of accuracy given up, the encoder saves about as much as always using Flash does,
and less than that at the 1, 2 and 5 point limits.

### Is the router more than benchmark recognition?

Shuffling the router's choices inside each benchmark, keeping the number sent to Flash
(2,000 shuffles):

| Tuned for | Router: saved / lost | Shuffled: saved / lost (95% range of loss) | Router's accuracy advantage | Shuffles at least as accurate |
|---|---|---|---:|---:|
| 1 pp | 4.9% / 1.26 pp | 5.9% / 1.89 pp (1.21 to 2.57) | +0.63 pp | 4.0% |
| 2 pp | 12.9% / 2.96 pp | 14.4% / 3.92 pp (3.16 to 4.71) | +0.96 pp | 1.1% |
| 3 pp | 20.1% / 4.03 pp | 24.3% / 5.76 pp (4.85 to 6.60) | +1.73 pp | 0.0% |
| 5 pp | 28.0% / 6.31 pp | 34.1% / 7.71 pp (6.84 to 8.64) | +1.40 pp | 0.1% |

As with logistic regression, the encoder picks better individual questions than policies
with the same per-benchmark routing rates.

### If the workload had more coding

| Coding share | Saved / lost at 1 pp target | Saved / lost at 5 pp target |
|---|---:|---:|
| 10% | 5.0% / 1.27 | 28.1% / 6.33 |
| 25% | 3.6% / 1.05 | 21.0% / 5.43 |
| 50% | 2.0% / 0.70 | 12.2% / 3.94 |

### Sensitivity checks

| Check | Bill saved (1 pp target) | Accuracy lost, pp |
|---|---:|---:|
| Main result | 4.9% | 1.26 |
| HLE with repaired costs (2,167 questions) | 4.3% | 1.20 |
| Costs at list prices | 4.9% | 1.26 |

### Router overhead

Fitting took 8.0 minutes and prediction about 6.3 ms per question on a Tesla T4 GPU (the
report's hardware column shows the host CPU, because the final-evaluation script does not
record the GPU). The model weights are about 600 MB.

### Summary for the text encoder

- Tuned for a 1-point loss, the router cut the bill by 4.9% and lost 1.26 points.
- Tuned for 2 points: 12.9% saved for 2.96 points. 3 points: 20.1% for 4.03. 5 points:
  28.0% for 6.31.
- It overshot every accuracy target on the test, by more than logistic regression did.
- It sends questions from both SimpleQA and MMLU-Pro to Flash, and keeps coding and HLE on
  GPT-5.

## Logistic regression versus the text encoder

| Tuned for | Logistic: saved / lost | Encoder: saved / lost |
|---|---|---|
| 0 pp | 0.1% / 0.05 pp | 2.6% / 0.29 pp |
| **1 pp** | **5.7% / 1.17 pp** | **4.9% / 1.26 pp** |
| 2 pp | 10.3% / 1.99 pp | 12.9% / 2.96 pp |
| 3 pp | 16.1% / 3.20 pp | 20.1% / 4.03 pp |
| 5 pp | 25.8% / 5.10 pp | 28.0% / 6.31 pp |
| Saved per point lost (1 to 5 pp) | 4.9% to 5.2% | 3.9% to 5.0% |
| Mean validation error (development) | 0.2215 | 0.2215 |
| Fitting | 3.6 s on a laptop CPU | 8.0 min on a T4 GPU |

- At the main 1-point target the two are close, with logistic regression slightly ahead
  (more saved, less lost). Their intervals overlap.
- At the looser targets the encoder saves more, but only by sending more questions to
  Flash, and it loses correspondingly more accuracy: it overshoots each target by about a
  point. Per point of accuracy given up, logistic regression buys more saving.
- On the development data their average error is the same.
- Logistic regression is far simpler and cheaper to train and run.

On these results the fine-tuned ModernBERT matches but does not beat the TF-IDF logistic
regression. The final model is still named from the development results
(`docs/DEV_RESULTS.md`), not from this page.

## What these results do not cover

- Two models of three. Gradient boosting has not been tested, and the choice of final
  model is pending.
- These benchmarks in this mix. The workload tables show the saving depends on the mix.
- Intervals reflect which test questions were drawn, not refitting or re-tuning. The
  encoder was trained with one seed; its variation across seeds was not measured.
- The test questions were held out from all model fitting, tuning and selection. The
  earlier pilot and the data audit did touch them in aggregate.
