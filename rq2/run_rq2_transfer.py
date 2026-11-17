"""RQ2 pipeline: transfer to unseen kinds of questions.  ON HOLD: see rq2/README.md.

  python rq2/run_rq2_transfer.py <model> --plan           # row counts only; reads no outcomes (allowed now)
  python rq2/run_rq2_transfer.py <model> --dry-run        # refused until the design is settled
  python rq2/run_rq2_transfer.py <model> --unlock-final   # refused until the design is settled

The design coded below is a candidate, not an agreed plan. It trains one router on the source
pool only, tunes and freezes its cutoffs there, and scores it on the pool's own test rows
(reference) and on LiveCodeBench, AIME + LiveMathBench, and HLE.

Output (once enabled): results/rq2/<model>/
"""
import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from router import data as D, splits as S, registry, runlog, final

# The team sets this to True only after the RQ2 experiments are agreed and written into
# docs/PROTOCOL.md section 8.
DESIGN_SETTLED = False

args = sys.argv[1:]
pos_args = [a for a in args if not a.startswith("--")]
if len(pos_args) != 1:
    sys.exit(__doc__)
family = pos_args[0]
registry.check(family)
DRY, UNLOCK, PLAN = "--dry-run" in args, "--unlock-final" in args, "--plan" in args
if DRY + UNLOCK + PLAN != 1:
    sys.exit("choose exactly one of --plan, --dry-run or --unlock-final")

POOL = tuple(D.SELECTION_POOL)
not_in_pool = lambda r: not S.in_source_pool(r)
TARGETS = {
    "code": dict(priority="primary transfer", benchmarks=("livecodebench",)),
    "math": dict(priority="secondary transfer", benchmarks=("aime", "livemathbench")),
    "hle": dict(priority="secondary transfer", benchmarks=("hle",)),
}
ALL_SPLITS = ("train", "val", "test")

if PLAN:
    c = S.plan_counts
    tot = lambda d: f"{sum(d.values()):5d}  {d}"
    print("RQ2 candidate design (one router trained on the source pool only). NOT AGREED, NOT RUN.")
    print("  fit       ", tot(c(splits=("train",), benchmarks=POOL, exclude=not_in_pool)))
    print("  cutoffs   ", tot(c(splits=("val",), benchmarks=POOL, exclude=not_in_pool)))
    print("  reference: source-pool locked test", tot(c(splits=("test",), benchmarks=POOL, exclude=not_in_pool)))
    for name, t in TARGETS.items():
        print(f"  target {name} ({t['priority']})", tot(c(splits=ALL_SPLITS, benchmarks=t["benchmarks"])))
    print("  target hle, costs repaired", tot(c(splits=ALL_SPLITS, benchmarks=("hle",), use_recovered_cost=True)))
    sys.exit(0)

if not DESIGN_SETTLED:
    sys.exit("RQ2 is on hold: the team has not decided which RQ2 experiments to run. "
             "Nothing was read or run. See rq2/README.md.")

OUT = D.ROOT / "results" / ("rq2_dryrun" if DRY else "rq2") / family
RESULT_FILE = OUT / "final_results_rq2.json"
if UNLOCK and RESULT_FILE.exists():
    sys.exit(f"{RESULT_FILE.relative_to(D.ROOT)} already exists: this runs once per model.")
dev_path = D.ROOT / "results" / "rq2_dev_selection" / family / "summary.json"
if not dev_path.exists():
    sys.exit(f"{dev_path.relative_to(D.ROOT)} not found: run  python scripts/05_dev_compare.py {family} --source-pool  first.")
config = json.loads(dev_path.read_text())["final_config"]
print(f"model: {family}  study: RQ2  settings: {config}  mode: {'DRY RUN' if DRY else 'FINAL'}", flush=True)
OUT.mkdir(parents=True, exist_ok=True)
man = S.read_manifest()
results = dict(mode="dry_run" if DRY else "final", study="rq2", family=family, label=registry.label(family), config=config,
               dev_summary_sha256=D.sha256_file(dev_path), protocol_sha256=runlog.protocol_sha256(),
               env=runlog.env_info(), studies={})

if DRY:
    # Rehearsal on source-pool development rows only: pretend GPQA is an unseen benchmark.
    tab, folds = S.load_selection_dev("rq2_dry_run")
    rows_all = {r["qid"]: r for r in D.read_paired()}
    def sub(mask): return tab.take(np.flatnonzero(mask)), [rows_all[q] for q in tab.qid[mask]]
    notg = tab.benchmark != "gpqa"
    tr, _ = sub((folds <= 2) & notg); va, _ = sub((folds == 3) & notg)
    results["studies"]["rq2_transfer_rehearsal"] = final.run_one(
        family, config, "rq2_transfer_rehearsal", tr, va,
        {"pseudo_reference": sub((folds == 4) & notg), "gpqa_as_holdout": sub(tab.benchmark == "gpqa")}, OUT, man)
else:
    kw = dict(unlock_final=True)
    rep = dict(unlock_final=True, use_recovered_cost=True)  # HLE cost sensitivity: scoring only
    # The source pool's train and validation rows are development data: no unlock is needed to read them.
    tr, _ = S.load_rows("final:rq2:train", splits=("train",), benchmarks=POOL, exclude=not_in_pool)
    va, _ = S.load_rows("final:rq2:val", splits=("val",), benchmarks=POOL, exclude=not_in_pool)
    held_out = {b for t in TARGETS.values() for b in t["benchmarks"]}
    assert not (set(tr.benchmark.tolist()) | set(va.benchmark.tolist())) & held_out
    targets = {"source_pool_locked_test": S.load_rows("final:rq2:pool_test", splits=("test",), benchmarks=POOL, exclude=not_in_pool, **kw)}
    for name, t in TARGETS.items():
        targets[name] = S.load_rows(f"final:rq2:target:{name}", splits=ALL_SPLITS, benchmarks=t["benchmarks"], **kw)
    targets["hle_costs_repaired"] = S.load_rows("final:rq2:target:hle_repaired", splits=ALL_SPLITS, benchmarks=("hle",), **rep)
    results["studies"]["rq2_transfer"] = dict(final.run_one(family, config, "rq2_transfer", tr, va, targets, OUT, man),
                                              priority={n: t["priority"] for n, t in TARGETS.items()})

RESULT_FILE.write_text(json.dumps(results, indent=2))
print("wrote", RESULT_FILE.relative_to(D.ROOT))
