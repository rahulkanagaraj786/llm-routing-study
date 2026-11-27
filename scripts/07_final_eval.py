"""Final evaluation for RQ1 (the main study), for ONE model.

  python scripts/07_final_eval.py <model> --plan           # row counts; reads no outcomes
  python scripts/07_final_eval.py <model> --dry-run        # rehearsal on development rows only
  python scripts/07_final_eval.py <model> --unlock-final   # the real thing: ONCE per model

<model> is a name from src/router/registry.py. The model's settings are taken from its own
development run (results/dev_selection/<model>/summary.json), so that run must exist first.

What it does: trains the router on the train split of all seven benchmarks, tunes the
cutoffs for 0, 1 and 2 points on the validation split, freezes them, then scores the locked
test split once. It also reports the within-benchmark shuffle diagnostic, the workload-mix
check, and the same test with repaired HLE costs (docs/PROTOCOL.md sections 7 and 9).

RQ2 (transfer to unseen benchmarks) is not in this script: see rq2/run_rq2_transfer.py and docs/PROTOCOL.md section 8.

Output: results/final/<model>/final_results_rq1.json, predictions, cutoffs and the fitted model.
Option: --with-arc-agi adds the optional ARC-AGI check.
"""
import sys, json, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from router import data as D, splits as S, registry, runlog, final

args = sys.argv[1:]
pos_args = [a for a in args if not a.startswith("--")]
if len(pos_args) != 1:
    sys.exit(__doc__)
family = pos_args[0]
registry.check(family)
DRY, UNLOCK, PLAN = "--dry-run" in args, "--unlock-final" in args, "--plan" in args
WITH_ARC = "--with-arc-agi" in args
MIX_FOCUS, MIX_SHARES = "livecodebench", (0.10, 0.25, 0.50)  # workload-mix sensitivity, RQ1 only
if DRY + UNLOCK + PLAN != 1:
    sys.exit("choose exactly one of --plan, --dry-run or --unlock-final")
STUDY = "rq1"
CORPUS = tuple(D.CORPUS_BENCHMARKS)
if PLAN:
    c = S.plan_counts
    tot = lambda d: f"{sum(d.values()):5d}  {d}"
    print("RQ1 main study (router trained on all seven benchmarks)")
    print("  fit       ", tot(c(splits=("train",), benchmarks=CORPUS)))
    print("  cutoffs   ", tot(c(splits=("val",), benchmarks=CORPUS)))
    print("  test      ", tot(c(splits=("test",), benchmarks=CORPUS)))
    print("  test, HLE costs repaired", tot(c(splits=("test",), benchmarks=CORPUS, use_recovered_cost=True)))
    if WITH_ARC:
        print("  optional ARC-AGI", tot(c(splits=("fresh_sealed",), benchmarks=("arc-agi",))))
    sys.exit(0)

OUT = D.ROOT / "results" / ("final_dryrun" if DRY else "final") / family
RESULT_FILE = OUT / f"final_results_{STUDY}.json"
if UNLOCK and RESULT_FILE.exists():
    sys.exit(f"{RESULT_FILE.relative_to(D.ROOT)} already exists: the final evaluation runs once per model. "
             "Do not rerun it to compare alternatives.")
dev_path = D.ROOT / "results" / "dev_selection" / family / "summary.json"
if not dev_path.exists():
    sys.exit(f"{dev_path.relative_to(D.ROOT)} not found: run  python scripts/05_dev_compare.py {family}  first. "
             "The final evaluation uses the settings chosen there.")
config = json.loads(dev_path.read_text())["final_config"]
sel_path = D.ROOT / "results/dev_selection/selection.json"
print(f"model: {family}  study: RQ1  settings: {config}  mode: {'DRY RUN' if DRY else 'FINAL'}", flush=True)
OUT.mkdir(parents=True, exist_ok=True)
man = S.read_manifest()


results = dict(mode="dry_run" if DRY else "final", study=STUDY, family=family, label=registry.label(family), config=config,
               dev_summary_sha256=D.sha256_file(dev_path), protocol_sha256=runlog.protocol_sha256(),
               selection_sha256=D.sha256_file(sel_path) if sel_path.exists() else None,
               env=runlog.env_info(), with_arc_agi=WITH_ARC, studies={})

if DRY:
    # Rehearsal on development rows only (all seven benchmarks). Folds 0-2 train, 3 calibrate, 4 pseudo-test.
    tab, folds = S.load_rq1_dev("final_eval_dry_run")
    rows_all = {r["qid"]: r for r in D.read_paired()}
    def sub(mask): return tab.take(np.flatnonzero(mask)), [rows_all[q] for q in tab.qid[mask]]
    tr, _ = sub(folds <= 2); va, _ = sub(folds == 3)
    results["studies"]["rq1"] = final.run_one(family, config, "rq1", tr, va, {"pseudo_test": sub(folds == 4)}, OUT, man,
                                              mix_targets=("pseudo_test",), mix=(MIX_FOCUS, MIX_SHARES))
else:
    kw = dict(unlock_final=True)
    rep = dict(unlock_final=True, use_recovered_cost=True)  # HLE cost sensitivity: scoring only
    tr, _ = S.load_rows(f"final:rq1:{family}:train", splits=("train",), benchmarks=CORPUS, **kw)
    va, _ = S.load_rows(f"final:rq1:{family}:val", splits=("val",), benchmarks=CORPUS, **kw)
    targets = {"locked_test": S.load_rows(f"final:rq1:{family}:test", splits=("test",), benchmarks=CORPUS, **kw),
               "locked_test_hle_costs_repaired": S.load_rows(f"final:rq1:{family}:test:hle_repaired", splits=("test",), benchmarks=CORPUS, **rep)}
    if WITH_ARC:
        targets["optional_arc_agi"] = S.load_rows(f"final:optional:{family}:arc", splits=("fresh_sealed",), benchmarks=("arc-agi",), **kw)
    results["studies"]["rq1"] = dict(final.run_one(family, config, "rq1", tr, va, targets, OUT, man,
                                                   mix_targets=("locked_test",), mix=(MIX_FOCUS, MIX_SHARES)),
                                     priority="main study")

if not DRY:
    # Copy the record of test reads into the result file itself, so it travels with the
    # results in git. "this_run" = the reads just made; "earlier_on_this_machine" = any
    # previous test reads in this machine's log (a rerun after deleting results shows here).
    lines = S.ACCESS_LOG.read_text().splitlines() if S.ACCESS_LOG.exists() else []
    test_reads = [ln for ln in lines if "unlock_final=True" in ln and "splits=test" in ln]
    mine = [ln for ln in test_reads if f":{family}:" in ln.split("\t")[1]]
    this_run = mine[-2:] if len(mine) >= 2 else mine          # the test and its HLE-repaired version
    results["test_reads"] = dict(this_run=this_run, earlier_on_this_machine=mine[:-len(this_run)] if this_run else [],
                                 other_test_reads_on_this_machine=len(test_reads) - len(mine))
RESULT_FILE.write_text(json.dumps(results, indent=2))
print("wrote", RESULT_FILE.relative_to(D.ROOT))
