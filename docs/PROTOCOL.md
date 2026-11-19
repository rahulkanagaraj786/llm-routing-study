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
| ModernBERT (later the same day) | Not refitted on this Mac. Deferred to a GPU run |
| Final selection and evaluation (later the same day) | On hold. The team will wait for the ModernBERT GPU run, then apply the selection rule to all three candidates. The final evaluation is not to be run until the team says so; the design is still under discussion |

### Team decisions recorded on 2026-11-16, review round

| Item | Decision |
|---|---|
| Duplicates at the RQ2 boundary | A source-pool question whose near-duplicate group contains a question from a held-out benchmark is removed from the source pool, and the affected fits are rerun |
| Selection rule | Unchanged: 1 pp cutoff target, 1.5 pp eligibility cap on out-of-fold loss. No safety margin is added to the primary analysis; the 0 pp setting shows a more conservative policy |
| ModernBERT | One comparable run on the source pool, same folds, on a GPU: 1,024 tokens, learning rates 2e-5 and 5e-5, one seed. If GPU time falls through, the comparison is reported as CPU-only and the first ModernBERT run is cited as pilot evidence |
| RQ1 and RQ2 | Kept separate. RQ1 has its own fit on all seven benchmarks. RQ2 uses one source-only router, scored on the locked source test as a reference and on each held-out benchmark |
| Within-benchmark shuffle | Added as a diagnostic (section 9) |
| Workload mix | Added as a sensitivity check on the frozen RQ1 predictions (section 9) |
| Cost-aware routing | Future work |
| GPT-5 label | "LLMRouterBench gpt-5; reasoning effort was not recorded for the study benchmarks." |
| Status wording | "No router from the revised protocol has been evaluated on the locked splits." This acknowledges that the pilot and the corpus audit did touch those questions |

### Team decisions recorded on 2026-11-16, roles

| Item | Decision |
|---|---|
| Model pair | GPT-5 and Gemini 2.5 Flash, unchanged |
| Candidate models | Regularized logistic regression, gradient boosting, and a fine-tuned text encoder (ModernBERT or similar). Random forest is no longer a candidate; its development runs are kept as a record |
| Owners | One owner per model. Each owner writes their own model code against the shared contract (`src/router/base.py`) and runs it through the shared scripts (`docs/TEAM_GUIDE.md`) |
| Shared pipeline | Data, splits, cutoff tuning, evaluation and reporting are common to all models and are not changed per model |
| Testing | Every candidate is evaluated on the locked splits, once, with the settings fixed by its own development run |

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
| MMLU-Pro, filtered | 1,506 | 517 | 510 | 290 / 80 / 89, plus 1 boundary duplicate (train) |
| GPQA | 119 | 39 | 40 | none |
| **Source pool** | **4,210** | **1,430** | **1,406** | |

Boundary duplicate: one MMLU-Pro chemistry question (train split) is the same question as
an HLE item. It is removed from the source pool because HLE is a held-out target. A check
of every LiveCodeBench, AIME, LiveMathBench and HLE question against the source pool
found no other near-duplicate (the next closest pair has cosine 0.63 and is two different
questions).

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

Candidates from 2026-11-16 (this replaces the table above):

| Candidate (`name`) | Owner | Search grid | Status |
|---|---|---|---|
| Regularized logistic regression (`logistic`) | Rahul | C ∈ {0.1, 0.3, 1, 3, 10} × statistics {off, on} = 10 | development run done |
| Gradient boosting (`gbm`) | Pravalhika Kurapati | max_depth ∈ {3, 5} × learning_rate ∈ {0.05, 0.1} × statistics {off, on} = 8; fixed n_estimators=200, min_samples_leaf=20, subsample=0.8, max_features="sqrt", seed 221 | development run and RQ1 test done (log entry 18) |
| Fine-tuned text encoder (`encoder`) | Priyanka | learning rate ∈ {2e-5, 5e-5} = 2, chosen in rotation 0 and reused (see below) | development run and RQ1 test done (log entries 16 and 17) |

Each owner writes the grid into this table before their first full development run and
does not change it afterwards.

Settings of the text encoder as used in its development run and RQ1 test
(`src/router/encoder.py`, `ENCODER_GRID` and constructor defaults). This paragraph replaces
an earlier version written before those runs; see log entry 16 for what changed and when.
`answerdotai/ModernBERT-base` at revision `8949b909ec900327062f0ebf497f51aef5e6f0c8`, encoder
and three-class head fine-tuned together with cross-entropy, gain = P(+1) − P(−1). Prompts
truncated to 1,024 tokens. Batches of 16 prompts grouped by length (shuffled, cut into
chunks of 50 batches, sorted by length within a chunk, batch order shuffled), each computed
in micro-batches of at most 4,096 padded tokens with gradients accumulated, so the update is
that of the full batch (a micro-batch that still runs out of GPU memory is split in half and
retried). AdamW, weight decay 0.01 (none on biases and norms), 10% linear warm-up then linear
decay, gradient clipping at 1.0. Up to 3 epochs; the calibration fold is scored four times
per epoch, the best snapshot is kept, and training stops after three checks in a row without
improvement. Mixed precision on CUDA (bf16 where native, fp16 with loss scaling otherwise),
fp32 elsewhere. Seed 221, one seed. Both runs used a Google Colab Tesla T4 in fp16. GPU
kernels are not bit-for-bit deterministic, so a rerun can differ slightly in the last digits.

## 6. Development comparison and algorithm selection

> **Changed on 2026-11-16 (amendment 9).** RQ1 now has its own development run on the
> train+validation questions of **all seven benchmarks** (8,307 questions), so that each
> model's settings and the choice of final model reflect the mix RQ1 is about. The
> procedure below (five rotations, the selection rule) is unchanged; only the questions
> it runs on differ. The earlier source-pool development run described in the "Data" and
> "History" paragraphs is kept as RQ2's development run (`results/rq2_dev_selection/`).
> Test questions are never used in either.

Data: the train+val rows of the source pool, 5,640 questions (SimpleQA 3,459, filtered
MMLU-Pro 2,023, GPQA 158), in the 5 folds already recorded in the manifest. The 370
MMLU-Pro math and computer-science development rows and the one boundary duplicate are
removed before any preprocessing or model is fitted.

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
3. With that configuration, choose cutoffs for 0, 1 and 2 pp (and 3 and 5 pp, log entry 12) on the calibration fold
   (minimum recorded cost subject to accuracy ≥ always-GPT-5 accuracy − allowance; ties
   → higher accuracy → higher cutoff). Cutoffs are midpoints between adjacent gains.
4. Apply the frozen cutoffs to the evaluation fold.

ModernBERT compute bound: both learning rates are fit in rotation 0 only; the one with
the lower calibration MSE is reused in rotations 1–4 (6 fine-tunes instead of 10). This
lets one binary choice depend on fold 1, which is the evaluation fold of rotation 1;
disclosed as a small leak. Early stopping uses the calibration fold.

Pooling the five evaluation folds gives, per candidate, out-of-fold gross savings `S` and
accuracy loss `L` at each allowance, over 5,640 questions that the deciding policy never
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

Pending candidates. A model that has no development results yet is passed to
`scripts/06_select_algorithm.py` with `--defer`; the result is then provisional. The
selection becomes final when every candidate in `src/router/registry.py` has development
results, or the team records with `--drop` that one will not be run. The first ModernBERT
run on the unfiltered pool (archived) is pilot evidence only.

The selected family's configuration for all later fits is the one with the lowest mean
calibration MSE across rotations (for ModernBERT, the rotation-0 choice). Nothing is
retuned afterwards.

## 7. Main study, RQ1 (not run yet)

Fit the selected algorithm and frozen configuration on the train split of all seven
benchmarks (6,216), choose cutoffs for 0, 1, 2, 3, 5 pp on the validation split (2,091), freeze,
then evaluate once on the locked test split (2,060) with `scripts/07_final_eval.py
<model> --unlock-final`. Baselines: always GPT-5 and always Flash on the same questions.

Every candidate is evaluated this way, once, by its owner, using the settings chosen in
its own development run. A model's development results must be committed before its
final evaluation, and nothing about the model may change afterwards. The selection rule
of section 6 names the primary model from development results only; test results play no
part in that choice. The report leads with the primary model and shows the others
alongside.

HLE cost sensitivity (required): the same fitted router and the same frozen cutoffs are
scored a second time on the locked test split with the 499 repaired HLE rows admitted
(those that fall in the test split). Nothing is refitted or re-tuned for it.

Also reported for RQ1, from the same frozen decisions: the within-benchmark shuffle
diagnostic and the workload-mix sensitivity check (both defined in section 9).

## 8. Transfer to held-out benchmarks, RQ2 (not run yet)

> **On hold (2026-11-16).** The team has not yet decided which RQ2 experiments to run. The
> design below is one candidate and may change. Nothing in it is run by the current
> scripts; it has its own pipeline in `rq2/`, which refuses to run while on hold. Until RQ2 is
> settled, the three benchmarks named here stay out of model development, so that every
> option remains open.

Claim being tested: transfer to held-out coding and math benchmarks. The claim is about
benchmarks the router never saw, not about training data free of related material.

One router serves all three tests. It uses the selected algorithm and frozen
configuration, is trained on the source pool's train split (4,210 questions), and has its
cutoffs for 0, 1 and 2 pp tuned on the source pool's validation split (1,430). No
question from LiveCodeBench, AIME, LiveMathBench or HLE is used for training, validation,
algorithm selection or cutoff tuning. The router and cutoffs are frozen, then scored
separately on each held-out benchmark, using every eligible question of it.

| Priority | Target | Questions |
|---|---|---:|
| Reference | Source pool, locked test rows (same types as training) | 1,406 |
| Primary | Coding: LiveCodeBench | 1,055 |
| Secondary | Math: AIME (58) + LiveMathBench (117) | 175 |
| Secondary | HLE | 1,631 (2,130 in the cost sensitivity check) |

- The source-pool reference is the same frozen router and cutoffs scored on test
  questions of the types it was trained on. It shows what changes when that router
  transfers. It is not a second RQ1.
- The within-benchmark shuffle diagnostic (section 9) is reported for every target.
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
- Within-benchmark shuffle (diagnostic, not a baseline). Take the router's frozen
  decisions on an evaluation set. Inside each benchmark, shuffle which questions go to
  Flash, keeping the number sent to Flash in that benchmark unchanged. Repeat 2,000
  times (seed 221) and report the shuffled policies' mean accuracy and cost with 95%
  ranges, next to the router's own. A router that only learned benchmark-level routing
  rates looks like its shuffles; a router that picks better individual questions has
  higher accuracy than its shuffles at a similar cost.
- Workload-mix sensitivity (RQ1 only). Reweight the locked-test questions so that
  LiveCodeBench makes up 10%, 25% and 50% of the questions, keeping the other benchmarks
  in their observed proportions to each other, and recompute accuracy, cost, savings and
  loss for the frozen router and both baselines. No refitting and no new cutoff.
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
2. ~~Who runs ModernBERT on a GPU, and by when.~~ Resolved 2026-11-18: the text encoder
   was run by its owner on a Google Colab Tesla T4 (log entries 16 and 17).
3. The team's go-ahead to run `scripts/07_final_eval.py --unlock-final`.

Status: no router from the revised protocol has been evaluated on the locked splits.

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

5. 2026-11-16, final selection and evaluation on hold. The team chose to wait for the
   ModernBERT GPU run before fixing the algorithm, and not to run the final evaluation
   yet. The two-candidate result is provisional. No sealed data has been read.

6. 2026-11-16, review round, before any refit and with no router from the revised
   protocol evaluated on the locked splits. (a) One boundary duplicate removed from the
   source pool (5,641 to 5,640 development questions); the two CPU candidates are
   refitted and the earlier filtered run is archived. (b) RQ2 gains the locked source
   test as a reference target. (c) Within-benchmark shuffle diagnostic and workload-mix
   sensitivity added, both computed from frozen decisions. (d) ModernBERT budget and the
   CPU-only fallback fixed. (e) Selection rule confirmed unchanged; no safety margin.

7. 2026-11-16, roles and candidates. (a) Random forest dropped as a candidate and
   replaced by gradient boosting; the encoder candidate may be ModernBERT or a similar
   model. (b) Each model has one owner who writes its code against a shared contract.
   (c) Every candidate, not only the selected one, is evaluated on the locked splits,
   once, with settings fixed by its development run; the choice of primary model still
   uses development results only. (d) The shared scripts now take the model name as an
   argument and write to per-model folders. The logistic regression development results
   were regenerated with the reorganised code and are identical to the previous ones.
   No router from the revised protocol has been evaluated on the locked splits.

8. 2026-11-16, RQ2 on hold. The team is focusing on RQ1 and will decide the RQ2
   experiments later. The final-evaluation script now covers RQ1 only; RQ2 has a separate
   pipeline in `rq2/` that is disabled until the design is agreed. HLE, LiveCodeBench, AIME and LiveMathBench remain excluded from the
   development comparison in the meantime. No router from the revised protocol has been
   evaluated on the locked splits.

9. 2026-11-16, RQ1 development run on all seven benchmarks. Until now the development
   comparison ran on the source pool (three benchmarks) for both research questions, to
   keep the held-out benchmarks out of model choice. With RQ1 and RQ2 now separate
   pipelines, RQ1's development run uses the train+validation questions of all seven
   benchmarks, and RQ2 keeps the source-pool run as its own. Reason: the final model for
   RQ1 should be chosen on the kinds of questions RQ1 covers, including coding, math and
   HLE. Decided before any model was evaluated on the locked splits, and before the other
   two models were written. Logistic regression's development run is redone on the new
   set; its source-pool results are moved, not discarded.

10. 2026-11-16, first final evaluation. Logistic regression's RQ1 test was run once, at
    the team's instruction, from commit `7bd7a49` with the settings fixed by its
    development run. Results are in `results/final/logistic/` and `docs/RQ1_RESULTS.md`.
    From this point nothing about logistic regression, the splits, the cutoff rule or the
    measurements may change. The other two models have not been tested.

11. 2026-11-16, reproducibility check. Logistic regression's RQ1 test was repeated from a
    clean state with identical settings (results removed, caches cleared). Cutoffs and
    per-question predictions were byte-identical to the first run, and every reported
    value matched. At the team's request the latest clean-state run is kept as the result
    of record; it differs from the first only in run metadata (timings, timestamp, the
    hash of this document). No setting was changed, so these are repeats of the same
    evaluation, not new ones.

12. 2026-11-16, two more accuracy limits. Limits of 3 and 5 points were added to the
    original 0, 1 and 2, to show how the trade-off continues for a company willing to give
    up more accuracy. They were added AFTER logistic regression's RQ1 test result at 0, 1
    and 2 points had been seen, so for logistic regression they are not pre-declared: they
    are reported and labelled as added afterwards. They are tuned and scored exactly like
    the others (cutoff chosen on validation, frozen, scored on the test). For gradient
    boosting and the text encoder, which have not been tested, all five limits are declared
    in advance. The 1-point limit remains the primary one and the selection rule is
    unchanged. Logistic regression's development run and test are regenerated so that all
    five limits come from the same code; its 0, 1 and 2 results must come out unchanged.
    Done the same day: after the rerun, the development gains and decisions, and the test
    cutoffs and results at 0, 1 and 2 points, were identical to the previous record.

13. 2026-11-17, test reads for logistic regression. `logs/data_access.log` shows four reads
    of the locked test split for logistic regression, all with the same model settings:
    (1) the first run (entry 10); (2) a repeat from a clean state (entry 11); (3) a second
    clean-state repeat, kept at the team's request as the record (entry 11); (4) the rerun
    that added the 3 and 5 point limits (entry 12), which is the current record. Runs 1 to 3
    were byte-identical, and run 4 reproduced their 0, 1 and 2 point results exactly. No
    further test reads are planned for logistic regression.

14. 2026-11-17, robustness fix with no effect on results. `runlog.protocol_sha256()` now
    returns nothing instead of failing when `docs/PROTOCOL.md` is absent, so the shared
    scripts run on a copy of the repository that does not include this document. Where the
    file exists the behaviour is unchanged. No result was recomputed.

15. 2026-11-17, recording of test reads. From now on `scripts/07_final_eval.py` names the
    model in every logged read and copies the record of test reads into the result file
    (`test_reads`), so that it is committed with the results. The team guide adds rules for
    recording results, including never deleting results to rerun a test. No result was
    recomputed; logistic regression's existing result file predates this and its reads are
    recorded in entry 13.

16. 2026-11-18, text encoder development run (log times 2026-11-19, UTC). The encoder is
    ModernBERT-base, as planned on 2026-11-16 (section 1): 1,024 tokens, learning rates
    2e-5 and 5e-5, one seed, on a GPU. Differences from that plan and how they arose:
    (a) It ran on Google Colab (Tesla T4, fp16) instead of the course server: the server's
    GPU driver was too old for the pinned torch build, and most of its GPU nodes were down.
    (b) Section 5's settings paragraph was first written before any run, but the settings
    then changed during wiring checks (`--smoke`, a one-in-eight sample that prints one
    calibration value and writes nothing), which ran out of GPU memory and were slow: the
    batching became length-grouped batches computed in micro-batches of at most 4,096
    tokens, fp16 was used on the T4, and the calibration fold is scored four times per epoch
    with a patience of three checks instead of once per epoch with a patience of one. The
    paragraph was not updated at the time and is corrected now, after the results. Only
    smoke-check values on the sample had been seen when these changes were made; no
    full-run result existed. The learning-rate grid did not change. (c) The full run was
    interrupted twice and resumed from the saved rotation files; the rotation in progress
    each time (rotations 3 and 4) was refitted from the start with the same settings, and
    its partial fits were discarded. Learning rate 2e-5 was chosen in rotation 0
    (calibration error 0.2295 against 0.2305 for 5e-5). Mean calibration error 0.2215.
    Results: `results/dev_selection/encoder/`, `docs/DEV_RESULTS.md`. The selection rule
    (section 6) has not yet been applied to these results.

17. 2026-11-18, text encoder RQ1 final evaluation (log time 2026-11-19 04:42 UTC). Run once
    by its owner from commit `6442d62` with the settings fixed by its development run
    (learning rate 2e-5), after a dry run on development questions. Its result file records
    two reads of the locked test split, both in this one run (the main test and the HLE
    repaired-cost check), and no earlier encoder reads on that machine. All five accuracy
    limits (0, 1, 2, 3 and 5 points) were declared before this test. Results:
    `results/final/encoder/` and `docs/RQ1_RESULTS.md`. From this point nothing about the
    text encoder may change. Gradient boosting has not been tested.

18. 2026-11-19, gradient boosting RQ1 final evaluation. Run once by its owner from commit
    `42f0a8a` with the settings fixed by its development run (max_depth = 3,
    learning_rate = 0.05, without the simple prompt statistics), after a dry run on
    development questions and with the team's go-ahead to unlock. Its result file records
    two reads of the locked test split, both in this one run (the main test and the HLE
    repaired-cost check), and no earlier gradient-boosting reads of the test on that
    machine. All five accuracy limits (0, 1, 2, 3 and 5 points) were declared before this
    test. Results: `results/final/gbm/` and `docs/RQ1_RESULTS.md`. From this point nothing
    about gradient boosting may change.
