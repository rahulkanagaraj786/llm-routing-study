# RQ2: transfer to unseen kinds of questions

RQ2 asks: does the cost and accuracy trade-off still hold on kinds of questions the router
never saw in training? The design is in `docs/PROTOCOL.md` section 8. RQ2 evaluates logistic
regression only. Results of A and A2: `docs/RQ2_RESULTS.md`. Workstream B design: section 8
(log entry 25); not yet unlocked.

## Workstreams

| | Experiment | Owner | State |
|---|---|---|---|
| A | One router trained on the source pool only (SimpleQA, GPQA, MMLU-Pro without math and computer science), scored on LiveCodeBench (primary), AIME + LiveMathBench, HLE, and the pool's own test questions (reference). Compared with the RQ1 router on the same questions; ranking-versus-cutoff diagnostic | Rahul | run once, 2026-11-26 |
| A2 | The same recipe trained without MMLU-Pro law, history and philosophy: new topics (those subjects) against new kinds of task (coding, math, HLE) | Rahul | run once, 2026-11-26 |
| B | Simulated few-shot adaptation on LiveCodeBench: retune cutoff using `k` labelled LCB-train questions, or retrain with those labels; score LCB test; compare to A's frozen failure at `k=0` | Pravalhika | designed 2026-11-27 (overlap audit done); not yet unlocked |
| C | Shifting workloads, and warning signs that need no labels | Priyanka | to be designed |
| C2 | Send unfamiliar questions to GPT-5, with the threshold set on source data only | Priyanka | to be designed |

**Primary RQ2 result:** A's router on LiveCodeBench at the 3-point target level, next to
the same router on the source pool's test questions. Everything else is secondary. The
reading rule (within the target, clearly exceeded, clear savings) is fixed in section 8.

B, C and C2 reuse A's frozen router and its saved predictions
(`results/rq2/logistic/predictions_rq2_transfer__*.csv`) and must not change them. Each
owner writes their design into section 8 before running. B's own outputs go under
`results/rq2/logistic/b_fewshot/`.

## Running A and A2

    python rq2/run_rq2_transfer.py logistic --plan           # question counts; reads no outcomes
    python rq2/run_rq2_transfer.py logistic --dry-run        # rehearsal on source-pool development questions
    python rq2/run_rq2_transfer.py logistic --unlock-final   # the real run, ONCE
    python rq2/report_rq2.py logistic                        # tables and figures from the saved results
    python rq2/posthoc_prompt_statistics.py logistic         # post hoc: why coding goes to Flash (prompts only)

The rehearsal uses development questions only: GPQA plays the far target, MMLU-Pro biology
and health play the held-out subjects, and law, history and philosophy are left out, so
nothing about the real targets is seen. The real run fits both routers, freezes their
cutoffs, scores every set once, computes the comparisons and diagnostics from the frozen
decisions, and refuses to run a second time. Output: `results/rq2/logistic/`.

## Workstream B (designed; runner not unlocked yet)

Overlap audit (prompts only; done):

    python rq2/audit_rq2_b_overlap.py
    # -> results/rq2/logistic/b_fewshot/overlap_audit.json
    #    label_pool_qids.json (633), eval_qids.json (211)

Run (unlock only after team go-ahead):

    python rq2/run_rq2_fewshot.py logistic --plan
    python rq2/run_rq2_fewshot.py logistic --dry-run      # no sealed LCB test outcomes
    python rq2/report_rq2_b.py logistic --dry-run        # tables/figs from the rehearsal
    python rq2/run_rq2_fewshot.py logistic --unlock-final # ONCE; refuses if results exist
    python rq2/report_rq2_b.py logistic

Output: `results/rq2/logistic/b_fewshot/` (dry-run twin under `results/rq2_dryrun/logistic/b_fewshot/`).
Do not modify A's files in `results/rq2/logistic/`.

## Settings

The router's settings (C = 0.3 with the simple prompt statistics) come from RQ2's own
development run, which used the source pool only:

    python scripts/05_dev_compare.py logistic --source-pool   # results/rq2_dev_selection/logistic/

RQ1's development run uses all seven benchmarks; the two are independent.
