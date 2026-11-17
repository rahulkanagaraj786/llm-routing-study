# RQ2: transfer to unseen kinds of questions (on hold)

**Status: not to be run.** The team is focusing on RQ1 and has not yet decided which RQ2
experiments to do. This folder is a separate pipeline so that RQ1 can be finished without
touching it. `run_rq2_transfer.py` refuses to run until the design is settled.

RQ2 asks: does the cost and accuracy trade-off still hold on kinds of questions the router
never saw in training?

## The candidate design in the script

One option, already coded, not agreed:

1. Train one router on a "source pool" only: SimpleQA, GPQA, and MMLU-Pro without its math
   and computer-science questions.
2. Tune its cutoffs on that pool's validation questions and freeze them.
3. Score the frozen router on:
   - the source pool's locked test questions (same kinds as training, the reference point);
   - LiveCodeBench (coding);
   - AIME + LiveMathBench (math, only 175 questions);
   - HLE (very hard expert questions).

## Open questions for the team

- Is "train on three knowledge benchmarks, test on coding, math and HLE" the experiment we
  want, or should each benchmark be held out in turn?
- Should each owner run RQ2 for their model, or only the primary model?
- Is the comparison point the same router on familiar test questions, or the RQ1 router?
- Do we re-tune the cutoff on a small sample of the new kind of question, to show how much
  that helps a company?
- How much weight does math get, given it has only 175 questions?

## What stays shared with RQ1

- The data, the splits and the eligibility rules (`src/router/data.py`, `splits.py`).
- The models and their settings (`src/router/registry.py`, each model's development run).
- The "fit, freeze the cutoffs, score" step and every measurement (`src/router/final.py`,
  `policy.py`).

Until RQ2 is settled, HLE, LiveCodeBench, AIME and LiveMathBench stay out of the
development comparison, so that any of the designs above is still possible.

## When the design is settled

1. Write the agreed design into `docs/PROTOCOL.md` section 8.
2. Edit `run_rq2_transfer.py` to match it.
3. Set `DESIGN_SETTLED = True` in that file.
4. Rehearse with `--dry-run`, then run once per model with `--unlock-final`.
