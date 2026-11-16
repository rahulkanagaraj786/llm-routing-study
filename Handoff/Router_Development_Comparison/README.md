# LLM Routing: Balancing Cost and Answer Accuracy

Start with `docs/TEAM_DIRECTION.md`, then `docs/EXPERIMENT_PLAN.md`.
This package supersedes the earlier development ZIP. It is a reproducible CPU pilot plus the current team handoff, not a complete GPU training implementation.

## Current scope
Compare logistic regression, random forest, and fine-tuned ModernBERT in development. Select one primary algorithm using validation. Use that algorithm for the main cost/accuracy and transfer studies. Baselines are always GPT-5 and always Gemini 2.5 Flash. Random routing is outside the agreed experiment/reporting scope.

## Contents
- `data/paired_full.jsonl.gz`: exact saved paired source, before current eligibility filters.
- `data/source_provenance.json`: original source metadata.
- `scripts/run_cpu_pilot.py`: runnable logistic regression and random forest development comparison.
- `results/development_pilot.json`: historical observed results with the out-of-scope diagnostic removed. No outcomes recomputed or improved by that removal.
- `docs/TEAM_DIRECTION.md`: goals, predictions, models, decisions, and limits.
- `docs/EXPERIMENT_PLAN.md`: evaluation and selection protocol, including unresolved choices.
- `docs/MODERNBERT_SPEC.md`: GPU implementation specification; not executed code.
- `docs/CODEX_HANDOFF.md`: ordered implementation tasks.
- `docs/PILOT_NOTES.md`: results, limitations, and reproducibility notes.

## Run the existing CPU pilot
Use a clean Python environment compatible with the pinned packages (validated here with the installed environment).

```bash
python -m pip install -r requirements-cpu.txt
python scripts/run_cpu_pilot.py
```

Run from any directory. Output goes to `results/development_rerun.json`, preserving the historical result file. This script is a development diagnostic: it has no untouched test evaluation and is NOT the final experiment pipeline.

ModernBERT has not been trained in this package. No best algorithm has been established. Attach the professor proposal separately; it is not embedded here. The proposal's original methods section may need updating if ModernBERT is added.
