# LLM Routing: Balancing Cost and Answer Accuracy

Course project for STA 221 (Fall 2026).

A prompt-only router between GPT-5 and Gemini 2.5 Flash, evaluated on recorded
LLMRouterBench outcomes. The router sees a question, predicts how much more likely GPT-5
is to answer it correctly than Flash, and sends it to GPT-5 only when that predicted
advantage is above a cutoff. Everything is offline: nothing here calls an LLM.

- **RQ1.** How much can a router save compared with always using GPT-5, and what does
  that do to accuracy?
- **RQ2.** Does that trade-off hold on kinds of questions the router never saw? (Design
  still being decided; the current code covers RQ1.)

## Status

| Step | State |
|---|---|
| Paired data table rebuilt from the public release | done |
| Train / validation / test split, locked test split | done, frozen |
| Shared pipeline: loading, cutoff tuning, evaluation, reports | done |
| Regularized logistic regression (reference model) | development run and RQ1 test done |
| Gradient boosting, fine-tuned text encoder | to be written by their owners |
| RQ2 | on hold; separate pipeline in `rq2/`, switched off |

If you are adding or running a model, start with `docs/TEAM_GUIDE.md`.

## Layout

```
src/router/data.py             paired table, eligibility, prompt-only inputs
src/router/splits.py           duplicate groups, hash-based splits, guarded loaders
src/router/policy.py           cutoff selection, metrics, bootstrap, diagnostics
src/router/base.py             the contract every model follows (fit, predict_gain)
src/router/registry.py         the list of models and their owners
src/router/models.py           shared text features; logistic regression (reference model)
src/router/template_model.py   skeleton to copy for a new model
src/router/final.py            the shared "fit, freeze cutoffs, score" step
scripts/                       numbered in run order
rq2/                           separate pipeline for RQ2, on hold
manifests/                     split manifest and its hashes
data/                          instructions for building the data (not stored here)
tests/                         checks for the evaluation code and the data guards
docs/                          team guide
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then follow `data/README.md` to download and build the data (one time per machine), and
check everything works:

```bash
python tests/test_policy.py && python tests/test_guards.py && python tests/test_models.py
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
  cutoffs for 0, 1, 2, 3 and 5 points of accuracy loss on the validation split, then score
  the locked test split once.
- `python scripts/06_select_algorithm.py` compares the models on development results and
  names the primary one.

## Rules that keep the evaluation honest

- Routers receive prompt text only. Scores and costs live in a separate object.
- `router.splits.load_rows` refuses test rows without `unlock_final=True`, and every data
  access is logged in `logs/data_access.log`.
- Model settings and the choice of final model use development rows only (train and
  validation); the test split is read only by the final evaluation.
- The final evaluation runs once per model and refuses to run a second time.
- Baselines are always GPT-5 and always Flash.
