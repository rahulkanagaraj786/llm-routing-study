# LLM Routing: Balancing Cost and Answer Accuracy

STA 221, Fall 2026. Rahul Kanagaraj, Priyanka Ramachandran, Pravalhika Kurapati.

A prompt-only router between GPT-5 and Gemini 2.5 Flash, evaluated on recorded
LLMRouterBench outcomes. The router sees a question, predicts how much more likely GPT-5
is to answer it correctly than Flash, and sends it to GPT-5 only when that predicted
advantage is above a cutoff. Everything is offline: no LLM is called anywhere in this
repository.

- **RQ1.** How much can a router save compared with always using GPT-5, and what does
  that do to accuracy?
- **RQ2.** Does that trade-off hold on benchmarks the router never saw?

## Where things stand

| Step | State |
|---|---|
| Data audit, paired table rebuilt from the raw release | done |
| Split manifest, locked test split, sealed held-out benchmarks | done, frozen |
| Loader, evaluator, logistic regression / random forest / ModernBERT pipelines | done |
| Algorithm selection on development data | provisional: done for the two CPU candidates, waiting for ModernBERT on a GPU |
| Final evaluation (RQ1 and RQ2) | script and reporting ready, **on hold, not run** |

Read in this order:
1. `docs/PROTOCOL.md`: the evaluation design, the selection rule, and every change to it.
2. `docs/DATA_AUDIT.md`: what the data is and its known problems.
3. `docs/DEV_RESULTS.md`: development results and the selected algorithm.

## Layout

```
src/router/data.py         paired table, eligibility, prompt-only Table / Outcomes
src/router/splits.py       duplicate groups, hash-based splits, guarded loaders
src/router/policy.py       cutoff selection, fixed-policy metrics, bootstrap
src/router/models.py       logistic regression and random forest routers
src/router/modernbert.py   ModernBERT router (CUDA, Apple MPS or CPU)
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
tar -xzf data/raw/bench-release.tar.gz -C data/raw/extracted -T logs/selected_members.txt

.venv/bin/python scripts/01_build_paired.py   # must print sha256 b0c551f4...
.venv/bin/python scripts/00_audit.py
.venv/bin/python tests/test_policy.py && .venv/bin/python tests/test_guards.py
```

`scripts/02_make_splits.py` is not rerun: the manifest in `manifests/` is frozen, and the
loaders refuse to work if it or the data file changes.

## Development comparison

Uses only the train and validation rows of the source pool (SimpleQA, GPQA, and MMLU-Pro
without its math and computer-science questions).

```bash
.venv/bin/python scripts/05_dev_compare.py logistic     # about 1 minute
.venv/bin/python scripts/05_dev_compare.py forest       # about 1.5 minutes
.venv/bin/python scripts/06_select_algorithm.py --defer modernbert
.venv/bin/python scripts/08_plot_dev.py
```

### ModernBERT on a GPU

Not yet run on the source pool. On a machine with a CUDA GPU of at least 16 GB:

1. Clone the repository and run the setup above (the data hash must match), installing a
   CUDA build of PyTorch.
2. `python scripts/05_dev_compare.py modernbert`, or submit
   `scripts/run_modernbert_slurm.sbatch`. It is resumable and writes one file per rotation.
   On CUDA it uses 1,024-token prompts, batches of 16 and mixed precision.
3. Copy `results/dev_selection/modernbert/` back and run
   `python scripts/06_select_algorithm.py` (without `--defer`).

On an Apple-silicon Mac the same command runs on MPS with 512-token prompts. Set
`PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.5 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.4` first, or it can
push a 16 GB machine into swap.

## Final evaluation

One time only, after the team agrees to unlock the sealed data. It refuses to run while
`selection.json` still lists a deferred candidate.

```bash
.venv/bin/python scripts/07_final_eval.py --plan           # row counts of every fit and target; reads no outcomes
.venv/bin/python scripts/07_final_eval.py --dry-run        # rehearsal on development rows
.venv/bin/python scripts/07_final_eval.py --unlock-final   # RQ1 on the locked test split, RQ2 on the held-out benchmarks
.venv/bin/python scripts/09_report_final.py                # tables and figures in results/final/report/
```

- RQ1: router trained on all seven benchmarks, cutoffs tuned on validation, scored on the
  locked test split.
- RQ2: one router trained on the source pool only, cutoffs tuned on its validation rows,
  then scored separately on LiveCodeBench, AIME + LiveMathBench, and HLE.
- Options: `--pool-reference` also scores the RQ2 router on the locked test rows of its own
  pool; `--with-arc-agi` adds the optional ARC-AGI check.

## Rules that keep the evaluation honest

- Routers receive prompt text only. Scores and costs live in a separate object.
- `router.splits.load_rows` refuses sealed rows without `unlock_final=True`, and every
  data access is logged in `logs/data_access.log`.
- Algorithm selection and tuning read only development rows of the source pool.
- The final evaluation refuses to run twice.
- Baselines are always GPT-5 and always Flash. There is no random-routing baseline.
