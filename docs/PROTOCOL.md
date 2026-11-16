# Evaluation protocol (frozen before any new model fit)

Project: LLM Routing: Balancing Cost and Answer Accuracy (STA 221, Fall 2026).
Written 2026-11-15, after the data audit and before fitting any new model. Every file
under `results/dev_selection/` records the SHA-256 of this document as it stood when the
result was produced (`protocol_sha256`), and `logs/protocol_freeze.log` records the hash
and time at which it was frozen.

Later changes go in the deviation log at the end, with a reason.

## 1. Team decisions recorded on 2026-11-15

| Decision | Choice |
|---|---|
| Honest test design | Lock a one-time ~20% test split now; never used for selection or tuning |
| Algorithm selection rule | Primary allowance 1 pp; savings measured on development folds not used to choose the cutoff |
| RQ2 holdouts | Code (LiveCodeBench), math (AIME + LiveMathBench), HLE |
| ModernBERT compute | Try this Mac (Apple MPS) first; remote GPU if a fit is too slow |

### Team decisions recorded on 2026-11-16 (before any sealed data was read)

| Item | Decision |
|---|---|
| HLE zero-cost rows | Main result keeps the pilot rule (1,631 HLE questions). The repaired sample (2,130) is a required sensitivity check; only the cost is copied, the grade stays the main file's |
| Transfer wording | "Transfer to held-out coding and math benchmarks". No claim that training contains no related material |
| Transfer priority | LiveCodeBench is the primary transfer result; math is secondary (175 questions) |
| Parsimony guard | Kept, with a minimum improvement defined (section 6) |
| ARC-AGI / SWE-bench | Not a required check and not called fresh data. ARC-AGI v1 is an optional check after freezing; SWE-bench is not evaluated |
| Source pool (later the same day) | SimpleQA, GPQA and MMLU-Pro without its math and computer-science questions. Used for algorithm selection, training, validation and cutoff tuning of the transfer router |
| Algorithm selection | Rerun from scratch on the source pool: all three candidates refitted, cutoffs retuned, qualification rechecked. The first run is kept as a record |
| Transfer design | One frozen router, trained on the source pool only, evaluated separately on LiveCodeBench, AIME + LiveMathBench, and HLE |
| Additional analyses | On hold until the primary findings are presented |
| ModernBERT (later the same day) | Not refitted on this Mac. Deferred to a GPU run; the selection for the main study is made among logistic regression and random forest |

## 2. Data

Source: `NPULH/LLMRouterBench` at revision `0e5af1b8…`, `bench-release.tar.gz`
(sha256 `b79f8cde…c8f1`, matches the handoff provenance). The paired table is rebuilt from
the raw files by `scripts/01_build_paired.py` into `data/processed/paired_v2.jsonl.gz`.
For the seven corpus benchmarks it reproduces the handoff package's scores, costs and
eligibility for all 10,918 rows.

Model pair: raw `model_name` values `gpt-5` (strong) and `gemini-2.5-flash` (cheap). The
release keeps `gpt-5-chat` in separate directories; none of it is used.

Target: `Y = GPT-5 correct − Flash correct` ∈ {−1, 0, +1}.

Eligibility (same as the pilot): same prompt for both models (whitespace-insensitive),
both scores in {0, 1}, both responses non-empty, both recorded costs > 0.

| Benchmark | Role | Eligible | Train | Val | Locked test |
|---|---|---:|---:|---:|---:|
| SimpleQA | selection pool | 4,315 | 2,585 | 874 | 856 |
| MMLU-Pro | selection pool | 2,993 | 1,797 | 597 | 599 |
| GPQA | selection pool | 198 | 119 | 39 | 40 |
| HLE | reserved holdout | 1,631 | 976 | 336 | 319 |
| LiveCodeBench | reserved holdout (code) | 1,055 | 633 | 211 | 211 |
| AIME | reserved holdout (math) | 58 | 36 | 11 | 11 |
| LiveMathBench | reserved holdout (math) | 117 | 70 | 23 | 24 |
| **Corpus total** | | **10,367** | **6,216** | **2,091** | **2,060** |
| ARC-AGI v1 | unused, sealed; optional check | 396 | | | |
| SWE-bench Verified | unused, sealed; not evaluated | 491 | | | |

Source pool (after removing MMLU-Pro math and computer science, 459 eligible rows):

| Benchmark | Train | Val | Locked test | Removed (train / val / test) |
|---|---:|---:|---:|---|
| SimpleQA | 2,585 | 874 | 856 | none |
| MMLU-Pro, filtered | 1,507 | 517 | 510 | 290 / 80 / 89 |
| GPQA | 119 | 39 | 40 | none |
| **Source pool** | **4,211** | **1,430** | **1,406** | |

ARC-AGI and SWE-bench are paired runs from the same archive that the pilot did not use.
They are not newly collected data. (The manifest and code label them `fresh` /
`fresh_sealed`; that is an internal name kept so that the frozen hashes do not change.)

Predictor inputs: the prompt text only. Scores, costs, token counts, responses, ground
truth and benchmark identity are never predictor inputs. The code enforces this by keeping
outcomes in a separate object that featurizers and `predict_gain` never receive.

## 3. Splits

`scripts/02_make_splits.py` writes `manifests/split_manifest_v1.csv`
(sha256 `0f058161…a69c`); loaders refuse to run if the manifest or data hash changes.

- Near-duplicate groups: questions whose normalized question text has character 4-gram
  TF-IDF cosine ≥ 0.85 are grouped (160 groups, 381 rows, largest 5). A group never
  straddles splits. Grouping uses prompt text only.
- Assignment: within each benchmark, groups are ordered by SHA-256 of a fixed salt and the
  group id; the first 20% are test, the next 20% validation, the rest train. No random
  number generator is involved, and assignment is computed over all rows, so later
  eligibility decisions cannot move a question between splits.
- Five cross-validation folds are assigned the same way to train+val groups.

Sealed data: the locked test split, the three reserved holdout benchmarks, and the two
unused benchmarks (ARC-AGI, SWE-bench). `router.splits.load_rows` raises an error for
these unless called with `unlock_final=True`; every data access is appended to
`logs/data_access.log`.

## 4. Prior exposure (to be stated in the report)

- The handoff pilot fit untuned logistic regression and random forest on three random
  75/25 splits of the whole 10,367-question corpus and reported aggregate validation
  metrics. That pilot was rerun once on this machine before the test split existed.
- So every corpus question, including the locked test split and the holdout benchmarks,
  has contributed to aggregate pilot numbers. No per-question pilot output was inspected,
  and no hyperparameter was tuned on it.
- This audit computed per-benchmark accuracy and cost of the two models (baseline facts,
  not router results) on all rows, including ARC-AGI and SWE-bench.
- The locked test is therefore "held out from all new fitting, selection and tuning", not
  "never seen". ARC-AGI and SWE-bench were never in the pilot corpus.

## 5. Candidates and declared search budget

All three estimate the expected advantage `gain(x) ≈ E[Y | prompt]`.

| Candidate | Features | Head | Grid |
|---|---|---|---|
| Logistic regression | word 1–2 gram TF-IDF (min_df 3, top 12,000 terms, sublinear tf), optional 10 simple statistics | 3-class multinomial, gain = P(+1) − P(−1) | C ∈ {0.1, 0.3, 1, 3, 10} × statistics {off, on} = 10 |
| Random forest | same TF-IDF, optional statistics | squared-error regression on Y (as in the pilot) | pilot setting, plus 300 trees × leaf {5, 20} × max_features {0.1, 0.3} with statistics = 5 |
| ModernBERT-base | tokens (pinned revision `8949b909…`) | 3-class head, cross-entropy, gain = P(class 2) − P(class 0) | learning rate {2e-5, 5e-5}; ≤ 3 epochs with early stopping |

Notes:
- The comparison is between full pipelines (features + model), not classifiers alone.
- TF-IDF vocabulary ties are broken by term, not by an unstable sort. The pilot's
  `max_features` cut filled 388–497 of 12,000 slots arbitrarily, which is the most likely
  reason its numbers do not reproduce exactly across machines.
- ModernBERT local settings (measured to fit this 16 GB Mac under a 0.5 MPS memory
  watermark): prompts truncated to 512 tokens, ≤ 8 prompts and ≤ 2,048 padded tokens per
  batch, AdamW, weight decay 0.01, 10% warm-up, linear decay, fp32, seed 221. In the
  selection pool 512 tokens truncates 0% of SimpleQA, 3.3% of MMLU-Pro, 3.5% of GPQA.
- Seeds: 221 for the forest and ModernBERT. One seed each; seed variance is not estimated.

## 6. Algorithm selection (source pool only)

Data: the train+val rows of the source pool, 5,641 questions (SimpleQA 3,459, filtered
MMLU-Pro 2,024, GPQA 158), in the 5 folds already recorded in the manifest. The 370
MMLU-Pro math and computer-science development rows are removed before any
preprocessing or model is fitted.

History: the selection was first run on 2026-11-15 with those 370 rows included (6,011
questions) and chose logistic regression. That run is kept in
`results/dev_selection_v1_unfiltered/` and `docs/archive/`. By team decision it is
superseded by a full rerun on the source pool with the same grids, rule and settings.
The rerun happens after the first run's results were seen; the reason for it is the
design of the transfer study, not those results.

For rotation k = 0..4: evaluation fold k, calibration fold (k+1) mod 5, fit on the other
three folds.

1. Fit every grid configuration on the fit folds.
2. Choose the configuration with the lowest calibration-fold squared error between gain
   and Y ("advantage MSE", a proper score for E[Y | x]). Cost is not used here.
3. With that configuration, choose cutoffs for 0, 1 and 2 pp on the calibration fold
   (minimum recorded cost subject to accuracy ≥ always-GPT-5 accuracy − allowance; ties
   → higher accuracy → higher cutoff). Cutoffs are midpoints between adjacent gains.
4. Apply the frozen cutoffs to the evaluation fold.

ModernBERT compute bound: both learning rates are fit in rotation 0 only; the one with
the lower calibration MSE is reused in rotations 1–4 (6 fine-tunes instead of 10). This
lets one binary choice depend on fold 1, which is the evaluation fold of rotation 1;
disclosed as a small leak. Early stopping uses the calibration fold.

Pooling the five evaluation folds gives, per candidate, out-of-fold gross savings `S` and
accuracy loss `L` at each allowance, over 5,641 questions that the deciding policy never
used for fitting, configuration choice, or cutoff choice.

Selection rule (1 pp allowance):
1. A candidate qualifies if `L ≤ 1.5 pp` (allowance plus a 0.5 pp tolerance for
   out-of-sample noise). If none qualifies, report that, and take the smallest `L`.
2. Among qualified candidates the winner is the highest `S`.
3. Parsimony guard (kept by team decision on 2026-11-16). Order the routers from
   simplest to most complex: logistic regression, random forest, ModernBERT. Among
   qualified routers the simpler one is kept unless the more complex router's
   out-of-fold savings are higher by at least 1.0 percentage point and the 90%
   paired-bootstrap interval for that difference excludes zero. The comparison uses
   development questions that were not used to fit the router or tune its cutoff; the
   test split plays no part. "Simpler" and "more complex" describe the routing
   algorithm, not GPT-5 versus Flash. The strict "highest S" winner is reported alongside.

   The comparison was run on 2026-11-15 with the interval condition only. The 1.0-point
   minimum was written down the next day, after the results were known. It could not
   have changed the outcome: both more complex routers saved less than logistic
   regression. It applies to any later rerun of the selection.

Deferred candidate. ModernBERT was not refitted on the source pool: the refit on this
Mac was stopped after one epoch of its first fit, before it produced any prediction, and
the team deferred it to a GPU. The selection rule above is therefore applied to logistic
regression and random forest. ModernBERT's only completed comparison is the first run on
the unfiltered pool (archived), where it did not beat logistic regression. When the GPU
run exists it follows this section unchanged, except that on a CUDA device it uses 1,024
tokens, batches of 16 and mixed precision. If the final evaluation has already been run
by then, the main-study algorithm stays as selected and the ModernBERT result is reported
as a development comparison only; the final evaluation is not repeated for it.

The selected family's configuration for all later fits is the one with the lowest mean
calibration MSE across rotations (for ModernBERT, the rotation-0 choice). Nothing is
retuned afterwards.

## 7. Main study, RQ1 (not run yet)

Fit the selected algorithm and frozen configuration on the train split of all seven
benchmarks (6,216), choose cutoffs for 0, 1, 2 pp on the validation split (2,091), freeze,
then evaluate once on the locked test split (2,060) with `scripts/07_final_eval.py
--unlock-final`. Baselines: always GPT-5 and always Flash on the same questions. Only the
selected algorithm is evaluated on the test split.

HLE cost sensitivity (required): the same fitted router and the same frozen cutoffs are
scored a second time on the locked test split with the 499 repaired HLE rows admitted
(those that fall in the test split). Nothing is refitted or re-tuned for it.

## 8. Transfer to held-out benchmarks, RQ2 (not run yet)

Claim being tested: transfer to held-out coding and math benchmarks. The claim is about
benchmarks the router never saw, not about training data free of related material.

One router serves all three tests. It uses the selected algorithm and frozen
configuration, is trained on the source pool's train split (4,211 questions), and has its
cutoffs for 0, 1 and 2 pp tuned on the source pool's validation split (1,430). No
question from LiveCodeBench, AIME, LiveMathBench or HLE is used for training, validation,
algorithm selection or cutoff tuning. The router and cutoffs are frozen, then scored
separately on each held-out benchmark, using every eligible question of it.

| Priority | Held-out target | Questions |
|---|---|---:|
| Primary | Coding: LiveCodeBench | 1,055 |
| Secondary | Math: AIME (58) + LiveMathBench (117) | 175 |
| Secondary | HLE | 1,631 (2,130 in the cost sensitivity check) |

- Math is secondary because of its size: one question is 0.57 pp, so two net additional
  errors already exceed a 1 pp loss. It is reported with its interval and read as
  descriptive.
- The allowance is defined on the source pool's validation questions. On a held-out
  benchmark the loss is measured against always-GPT-5 on that benchmark.
- MMLU-Pro subjects are read from the prompt template ("Answer the following math
  question…"). The original HLE dataset has `category` and `raw_subject` fields; they
  are not needed because all of HLE is held out.
- HLE cost sensitivity (required): the same frozen router and cutoffs are scored again
  on all 2,130 HLE questions with repaired costs.

Optional check after freezing: the RQ1 router may be applied once to ARC-AGI v1
(`--with-arc-agi`), reported as an unused benchmark from the same archive. It is off by
default. SWE-bench Verified is not evaluated.

## 9. Metrics and uncertainty

- Accuracy = mean selected correctness. Cost = sum of selected recorded costs.
- Gross savings % = 100 × (1 − router cost / always-GPT-5 cost).
- Accuracy loss (pp) = 100 × (always-GPT-5 accuracy − router accuracy); negative = gain.
- Pooled, per benchmark, and the equal-benchmark average.
- 95% paired percentile bootstrap over duplicate groups, 2,000 resamples, seed 221,
  decisions fixed. It does not include refitting, cutoff re-selection or model selection.
- Test loss is reported even when it exceeds the allowance.
- Overhead: fit time, prediction time per question, hardware. Savings are labelled gross.
  Net savings need a stated serving-cost assumption and are reported separately.

## 10. Status of open items

Resolved on 2026-11-16: HLE costs, transfer wording and priority, parsimony guard,
ARC-AGI / SWE-bench (see the decisions table in section 1).

HLE details behind that decision:

| Eligibility rule | Pilot rule | With repaired costs |
|---|---:|---:|
| Positive cost for both models, same question | 1,631 | 2,130 |
| Requiring byte-identical prompts | 1,630 | 2,129 |

- The one-row difference is HLE index 1494. Flash's prompt contains four U+2028 line
  separator characters; with those removed it equals GPT-5's prompt exactly, and the
  ground truth is the same. It is the same question and stays in.
- 507 GPT-5 rows have cost 0.0; 499 have a positive recorded cost in `hle/subset_500`
  for the same prompt, response text and token counts. One of them (index 185) has a
  different grade in the two files; the main file's grade is used and only the cost is
  copied.

Still open:
1. LiveCodeBench Flash costs are 20–100% of the price schedule every other file follows
   (total $6.16 recorded vs $8.46 at $0.30/$2.50 per million tokens). Default: recorded
   costs are primary and a list-price sensitivity figure is reported.
2. GPT-5 reasoning effort is not recorded in the result files. Label plots "GPT-5
   (LLMRouterBench `gpt-5`)" unless the run configuration is confirmed.
3. The team's go-ahead to run `scripts/07_final_eval.py --unlock-final`.

## Deviation log

1. 2026-11-15, ModernBERT local batch settings. The first ModernBERT run (rotation 0,
   learning rate 2e-5) ran out of MPS memory early in epoch 2 and produced no
   predictions. The per-batch limit was lowered from 2,048 to 1,536 padded tokens, cached
   MPS memory is released every 25 steps and after prediction, and a batch that still
   runs out of memory is split in half. Nothing about the data, grid, selection rule or
   other candidates changed. The only number seen from the failed run was one
   calibration-fold value (epoch-1 advantage MSE 0.2408), which is not a selection
   statistic. The logistic and forest results were produced under the previous hash of
   this file (`8e5a3fa8…`); the only difference is this entry.

2. 2026-11-16, amendments from team decisions, made before any sealed data was read
   (`logs/data_access.log` has no unlocked access). (a) HLE cost repair becomes a
   required sensitivity check scored with frozen routers and cutoffs, replacing the
   earlier all-or-nothing switch. (b) Transfer study reworded and prioritised: coding
   primary, math and HLE secondary. (c) Parsimony guard reworded and given a 1.0-point
   minimum, after the comparison; outcome unaffected. (d) ARC-AGI becomes optional and
   SWE-bench is dropped; neither is described as fresh data. No fitted model, split,
   grid or cutoff rule changed.

3. 2026-11-16, second set of team decisions, made before any sealed data was read and
   before any model was refitted. (a) The source pool drops MMLU-Pro math and
   computer-science questions (370 development rows, 89 test rows). (b) Algorithm
   selection is rerun in full on the source pool; grids, rotation scheme, selection rule
   and ModernBERT settings are unchanged, and the first run is archived. (c) The transfer
   study uses one router trained on the source pool only, replacing the three
   separately trained instances of amendment 2. (d) The three additional analyses
   proposed on 2026-11-16 are not part of the protocol.

4. 2026-11-16, ModernBERT deferred. The refit of ModernBERT on the source pool was
   stopped at the team's request (one epoch of the first fit completed, no predictions
   written; the only number seen was one calibration-fold value, 0.2532). The selection
   is made among the two CPU candidates and records ModernBERT as deferred. The GPU
   settings for the later run are fixed in section 6.
