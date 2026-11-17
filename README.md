# LLM Routing: Balancing Cost and Answer Accuracy

STA 221, Fall 2026. Rahul Kanagaraj, Priyanka Ramachandran, Pravalhika Kurapati.

A prompt-only router between GPT-5 and Gemini 2.5 Flash, evaluated on recorded
LLMRouterBench outcomes. The router sees a question, predicts how much more likely GPT-5
is to answer it correctly than Flash, and sends it to GPT-5 only when that predicted
advantage is above a cutoff. Everything is offline: no LLM is called anywhere in this
repository.

- **RQ1.** How much can a router save compared with always using GPT-5, and what does
  that do to accuracy?
- **RQ2.** Does that trade-off hold on benchmarks the router never saw? (Design still
  being decided; the current code covers RQ1.)

## Where things stand

| Step | State |
|---|---|
| Data audit, paired table rebuilt from the raw release | done |
| Split manifest, locked test split, sealed held-out benchmarks | done, frozen |
| Shared pipeline: loader, cutoff tuning, evaluation, reports | done |
| Logistic regression (reference model) | development run and RQ1 test done (`docs/RQ1_RESULTS.md`) |
| Gradient boosting, fine-tuned text encoder | to be written by their owners (`docs/TEAM_GUIDE.md`) |
| Final evaluation, RQ1 | script and reporting ready; no router from the revised protocol has been evaluated on the locked splits |
| RQ2 | design on hold; separate pipeline in `rq2/`, disabled until the team agrees the experiments |

Read in this order:
0. `docs/TEAM_GUIDE.md`: if you are adding or running a model, start here.
1. `docs/PROTOCOL.md`: the evaluation design, the selection rule, and every change to it.
2. `docs/DATA_AUDIT.md`: what the data is and its known problems.
3. `docs/DEV_RESULTS.md`: development results so far (logistic regression).
4. `docs/RQ1_RESULTS.md`: RQ1 test results, one section per model.

## Layout

```
src/router/data.py         paired table, eligibility, prompt-only Table / Outcomes
src/router/splits.py       duplicate groups, hash-based splits, guarded loaders
src/router/policy.py       cutoff selection, fixed-policy metrics, bootstrap
src/router/base.py         the contract every model follows (fit, predict_gain)
src/router/registry.py     the list of models and their owners
src/router/models.py       shared text features; logistic regression (the reference model)
src/router/template_model.py   skeleton to copy for a new model
src/router/final.py        the shared "fit, freeze cutoffs, score" step
rq2/                       separate pipeline for RQ2, on hold
scripts/                   numbered in run order, 00 to 09
manifests/                 split manifest and its hashes
results/audit              audit numbers, token lengths
results/dev_selection      out-of-fold predictions, cutoffs, selection.json
results/final              created by the final evaluation
docs/                      protocol, audit, development results
tests/                     evaluator and sealed-data guard checks
Handoff/                   proposal and the original pilot package, unchanged
```

## Setup

```bash
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python -r requirements.txt   # exact versions: requirements-lock.txt

# raw data: 1.28 GB, public, no account needed
mkdir -p data/raw/extracted
curl -L -o data/raw/bench-release.tar.gz \
  "https://huggingface.co/datasets/NPULH/LLMRouterBench/resolve/0e5af1b84bf73437a01a1849c0f1d2468baa93fc/bench-release.tar.gz"
shasum -a 256 data/raw/bench-release.tar.gz   # b79f8cde1a6f029c2efa663a3a3b6f7748defb22341fe59f328cebef6648c8f1
tar -xzf data/raw/bench-release.tar.gz -C data/raw/extracted -T data/selected_members.txt

.venv/bin/python scripts/01_build_paired.py   # one-time; checks the table against the expected hash
.venv/bin/python scripts/00_audit.py
.venv/bin/python tests/test_policy.py && .venv/bin/python tests/test_guards.py
```

`scripts/02_make_splits.py` is not rerun: the manifest in `manifests/` is frozen, and the
loaders refuse to work if it or the data file changes.

## Running a model

Every model goes through the same commands. `<model>` is a name from
`src/router/registry.py`, for example `logistic`.

```bash
python scripts/05_dev_compare.py <model> --smoke         # quick wiring check, writes nothing
python scripts/05_dev_compare.py <model>                 # development comparison (5 rotations)
python scripts/07_final_eval.py <model> --dry-run        # rehearsal on development questions
python scripts/07_final_eval.py <model> --unlock-final   # final evaluation, ONCE per model
python scripts/09_report_final.py <model>                # tables and figures
```

- The development comparison uses the train and validation questions of all seven
  benchmarks. It chooses each model's settings and gives the results from which the final
  model is named. It never touches the test split.
- The final evaluation answers RQ1: train on the train split of all seven benchmarks, tune
  cutoffs on the validation split, score the locked test split once.
- `python scripts/06_select_algorithm.py` compares the models on development results and
  names the primary one.

## Rules that keep the evaluation honest

- Routers receive prompt text only. Scores and costs live in a separate object.
- `router.splits.load_rows` refuses sealed rows without `unlock_final=True`, and every
  data access is logged in `logs/data_access.log`.
- Algorithm selection and tuning read only development rows of the source pool.
- The final evaluation runs once per model and refuses to run a second time.
- Baselines are always GPT-5 and always Flash. There is no random-routing baseline.
