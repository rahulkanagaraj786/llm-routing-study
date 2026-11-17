# Team guide: how to add and run your model

Everyone trains, validates and tests their model through the same pipeline, so the results
can be compared directly. You write one model file. Everything else is shared and fixed.

## Who owns what

| Part | Owner | Status |
|---|---|---|
| Shared pipeline: data, splits, cutoff tuning, evaluation, reports | Rahul | done |
| Regularized logistic regression (`logistic`) | Rahul | development run done; final test pending |
| Gradient boosting (`gbm`) | to be assigned | to be written |
| Fine-tuned text encoder, ModernBERT or similar (`encoder`) | to be assigned | to be written |

Logistic regression is the worked example. Read `LogisticRouter` in `src/router/models.py`
before writing yours.

## What is shared and what is yours

```
   SHARED, do not change                          YOURS
   ─────────────────────────────────────────      ──────────────────────────────────
   the data and which questions are used          one file: src/router/<your model>.py
   the train / validation / test split, 5 folds     - fit(prompts, y, ...)
   the label Y and the "question text only" rule     - predict_gain(prompts)
   how the cutoff is tuned (0, 1, 2 points)        your list of settings to try
   how savings and accuracy loss are measured      your entry in src/router/registry.py
   the scripts that run all of this                your results folder
```

## One-time setup

```bash
git clone https://github.com/rahulkanagaraj786/llm-routing-study.git
cd llm-routing-study
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Then build the data once by following `data/README.md` (a 1.3 GB download and one script).
Check everything works:

```bash
python tests/test_policy.py && python tests/test_guards.py && python tests/test_models.py
python scripts/05_dev_compare.py logistic --smoke
```

## Your workflow

### Step 1. Write your model

1. Copy `src/router/template_model.py` to `src/router/<name>.py` and rename the class.
2. Fill in two methods:
   - `fit(prompts, y, val_prompts, val_y)`: learn from the training questions.
   - `predict_gain(prompts)`: return one number per question between −1 and +1, the
     expected accuracy advantage of GPT-5 over Flash. For a three-class model this is
     `P(Y = +1) − P(Y = −1)`.
3. In `src/router/registry.py`, set `cls`, `grid` and `owner` in your model's entry.

Rules for the model code:
- Use only the question text. No scores, costs, answers or benchmark names.
- Fit all preprocessing on the training questions passed to `fit`, nothing else.
- Use `val_prompts` and `val_y` only for early stopping, if your model trains in epochs.
- Set a fixed random seed so reruns give the same numbers.

`TextFeatures` in `models.py` gives the same word-count features logistic regression uses,
if you want them.

### Step 2. Check it runs

```bash
python tests/test_models.py
python scripts/05_dev_compare.py <name> --smoke     # small sample, writes nothing
```

### Step 3. Fix your settings before the real run

List the settings you will try in your grid (a handful, not dozens) and write them into
the table in `docs/PROTOCOL.md` section 5. After you have seen real results, the grid must
not change. If something forces a change, add a line to the deviation log saying what and
why.

A slow model can set `tune_every_rotation=False` in the registry: the setting is then
chosen in the first rotation and reused, which cuts the number of fits.

### Step 4. Run the development comparison

```bash
python scripts/05_dev_compare.py <name>
```

It is resumable (one file per rotation). It trains on three folds, picks the best setting
and tunes the cutoffs on a fourth, and scores the fifth, five times over. It reads only
development questions and never the test split.

Output: `results/dev_selection/<name>/`. Commit that folder.

### Step 5. Rehearse the final evaluation

```bash
python scripts/07_final_eval.py <name> --plan        # shows the row counts, reads no outcomes
python scripts/07_final_eval.py <name> --dry-run     # rehearsal on development questions
python scripts/09_report_final.py <name> --dry-run   # tables and figures from the rehearsal
```

### Step 6. Run the final evaluation, once

This is the RQ1 test. Only after your development results are committed and the team has
agreed to go ahead:

```bash
python scripts/07_final_eval.py <name> --unlock-final
python scripts/09_report_final.py <name>
```

- It uses the settings chosen in step 4; you do not pick them again.
- It runs once per model and refuses to run a second time.
- After this, nothing about your model may change. A result you tuned after seeing the
  test is no longer a test result.

Output: `results/final/<name>/` with `final_results.json`, predictions, and a `report/`
folder of tables and figures. Commit it.

## What the final evaluation does for every model

It answers RQ1: how much does the router save compared with always using GPT-5, and what
does that do to accuracy?

- Trains on the train split of all seven benchmarks.
- Tunes the cutoffs for 0, 1 and 2 points on the validation split, then freezes them.
- Scores the locked test split once.
- Baselines on the same questions: always GPT-5 and always Flash.

RQ2 (unseen kinds of questions) is not part of this yet. The team is still deciding which
experiments to run for it. It has its own pipeline in `rq2/`, which is switched off until
the design is agreed; see `rq2/README.md` for the open questions.

## Do not change

- `src/router/data.py`, `splits.py`, `policy.py`, `final.py`, and anything in `manifests/`.
- The numbered scripts, except to fix a bug the whole team agrees on.
- Another person's model file or results folder.

If you think a shared piece is wrong, raise it with the team first. A change there affects
everyone's numbers.

## Working with git

- Work on a branch named after your model, for example `gbm`, and open a pull request.
- Commit your model file, your registry entry, and your two results folders.
- `data/raw/` and `data/processed/` are ignored on purpose; the data never goes in git.

## Checklist before you run step 6

- [ ] `tests/test_models.py` passes with your model.
- [ ] Your grid is written in `docs/PROTOCOL.md` and has not changed since your first real run.
- [ ] `results/dev_selection/<name>/` is committed.
- [ ] The dry run and its report look sensible.
- [ ] The team has agreed to run the final evaluations.

## How the models are compared at the end

Once every model has development results, Rahul runs `scripts/06_select_algorithm.py`. It
applies the rule in `docs/PROTOCOL.md` section 6 to the development results only and names
the primary model. Test results play no part in that choice, and the primary model is
never switched after test numbers are seen. The report then presents all three models'
test results, with the primary one leading.
