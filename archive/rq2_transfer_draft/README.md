# RQ2 transfer study: draft, on hold

`final_eval_with_rq2.py` is a snapshot of the final-evaluation script from when it covered
both research questions (`--rq1` / `--rq2`). The RQ2 part implements one possible design: a
router trained on SimpleQA, GPQA and filtered MMLU-Pro only, then scored on its own test
rows and on LiveCodeBench, AIME + LiveMathBench, and HLE.

The team has not yet decided which RQ2 experiments to run, so this is not part of the
pipeline and must not be run. When the design is settled, RQ2 gets its own script.
