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

RQ2 (transfer to unseen benchmarks) is not in this script. Its design is still being decided.

Output: results/final/<model>/final_results_rq1.json, predictions, cutoffs and the fitted model.
Option: --with-arc-agi adds the optional ARC-AGI check.
"""
import sys, json, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from router import data as D, splits as S, policy as P, registry, runlog

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


def fit_router(tr: D.Table, va: D.Table):
    return registry.make(family, config).fit(tr.prompts, tr.outcomes.y, va.prompts, va.outcomes.y)


def list_price_outcomes(rows) -> D.Outcomes | None:
    if any(r["strong_cost_list"] is None or r["cheap_cost_list"] is None for r in rows):
        return None
    return D.Outcomes(np.array([int(r["strong_score"]) for r in rows]), np.array([int(r["cheap_score"]) for r in rows]),
                      np.array([r["strong_cost_list"] for r in rows]), np.array([r["cheap_cost_list"] for r in rows]))


def report(gain, cutoffs, tab: D.Table, rows, label, mix=False):
    groups = np.array([man[q]["cluster_id"] for q in tab.qid])
    res = dict(label=label, n=len(tab), pp_per_question=100.0 / len(tab), allowances={},
               y_counts={str(v): int((tab.outcomes.y == v).sum()) for v in (-1, 0, 1)})
    lp = list_price_outcomes(rows)
    for a, c in cutoffs.items():
        d = P.route_strong(gain, c["cutoff"])
        m = P.evaluate(d, tab.outcomes)
        m.update(P.evaluate_by_benchmark(d, tab.outcomes, tab.benchmark))
        m["bootstrap"] = P.bootstrap(d, tab.outcomes, groups=groups)
        m["within_benchmark_shuffle"] = P.within_benchmark_shuffle(d, tab.outcomes, tab.benchmark)
        if mix and MIX_FOCUS in set(tab.benchmark.tolist()) and len(set(tab.benchmark.tolist())) > 1:
            m["workload_mix"] = [P.reweighted(d, tab.outcomes, tab.benchmark, MIX_FOCUS, sh) for sh in MIX_SHARES]
        m["cutoff"] = c["cutoff"]
        m["within_allowance"] = bool(m["accuracy_loss_pp"] <= float(a) + 1e-9)
        if lp is not None:
            ml = P.evaluate(d, lp)
            m["list_price_sensitivity"] = dict(gross_savings_pct=ml["gross_savings_pct"], cost=ml["cost"], always_strong_cost=ml["always_strong_cost"])
        res["allowances"][a] = m
    res["frontier"] = P.frontier(gain, tab.outcomes)
    np.savetxt(OUT / f"predictions_{label}.csv",
               np.column_stack([tab.qid, tab.benchmark, gain] + [P.route_strong(gain, c["cutoff"]).astype(int) for c in cutoffs.values()]),
               fmt="%s", delimiter=",", header="qid,benchmark,gain," + ",".join(f"route_gpt5_at_{a}pp" for a in cutoffs), comments="")
    return res


def run_one(label, tr, va, targets: dict, mix_targets=()):
    """Fit on tr, choose cutoffs on va, freeze, then evaluate each target table."""
    print(f"[{label}] fit on {len(tr)} rows, calibrate on {len(va)} rows", flush=True)
    r = fit_router(tr, va)
    g_va = r.predict_gain(va.prompts)
    cutoffs = {f"{a:g}": P.select_cutoff(g_va, va.outcomes, a) for a in P.ALLOWANCES_PP}
    (OUT / f"cutoffs_{label}.json").write_text(json.dumps(cutoffs, indent=2))  # frozen before any target is scored
    r.save(OUT / label / "model")
    cal = {a: P.evaluate(P.route_strong(g_va, c["cutoff"]), va.outcomes) for a, c in cutoffs.items()}
    out = dict(label=label, n_train=len(tr), n_calibration=len(va), fit_seconds=r.fit_seconds, cutoffs=cutoffs,
               calibration_metrics=cal, train_benchmarks=sorted(set(tr.benchmark.tolist())),
               calibration_benchmarks=sorted(set(va.benchmark.tolist())), targets={})
    for name, (tab, rows) in targets.items():
        t0 = time.perf_counter(); g = r.predict_gain(tab.prompts); dt = time.perf_counter() - t0
        res = report(g, cutoffs, tab, rows, f"{label}__{name}", mix=name in mix_targets)
        res["predict_ms_per_question"] = 1000 * dt / len(tab)
        out["targets"][name] = res
        m = res["allowances"]["1"]
        print(f"  {name}: n={len(tab)} | 1 pp: savings {m['gross_savings_pct']:.2f}% loss {m['accuracy_loss_pp']:.2f} pp to-Flash {100*m['cheap_fraction']:.1f}%", flush=True)
    return out


results = dict(mode="dry_run" if DRY else "final", study=STUDY, family=family, label=registry.label(family), config=config,
               dev_summary_sha256=D.sha256_file(dev_path), protocol_sha256=runlog.protocol_sha256(),
               selection_sha256=D.sha256_file(sel_path) if sel_path.exists() else None,
               env=runlog.env_info(), with_arc_agi=WITH_ARC, studies={})

if DRY:
    # Rehearsal on source-pool development rows only. Folds 0-2 train, 3 calibrate, 4 pseudo-test.
    tab, folds = S.load_selection_dev("final_eval_dry_run")
    rows_all = {r["qid"]: r for r in D.read_paired()}
    def sub(mask): return tab.take(np.flatnonzero(mask)), [rows_all[q] for q in tab.qid[mask]]
    tr, _ = sub(folds <= 2); va, _ = sub(folds == 3)
    MIX_FOCUS = "mmlupro"  # rehearsal only: there is no coding benchmark in the development rows
    results["studies"]["rq1"] = run_one("rq1", tr, va, {"pseudo_test": sub(folds == 4)}, mix_targets=("pseudo_test",))
else:
    kw = dict(unlock_final=True)
    rep = dict(unlock_final=True, use_recovered_cost=True)  # HLE cost sensitivity: scoring only
    tr, _ = S.load_rows("final:rq1:train", splits=("train",), benchmarks=CORPUS, **kw)
    va, _ = S.load_rows("final:rq1:val", splits=("val",), benchmarks=CORPUS, **kw)
    targets = {"locked_test": S.load_rows("final:rq1:test", splits=("test",), benchmarks=CORPUS, **kw),
               "locked_test_hle_costs_repaired": S.load_rows("final:rq1:test:hle_repaired", splits=("test",), benchmarks=CORPUS, **rep)}
    if WITH_ARC:
        targets["optional_arc_agi"] = S.load_rows("final:optional:arc", splits=("fresh_sealed",), benchmarks=("arc-agi",), **kw)
    results["studies"]["rq1"] = dict(run_one("rq1", tr, va, targets, mix_targets=("locked_test",)), priority="main study")

RESULT_FILE.write_text(json.dumps(results, indent=2))
print("wrote", RESULT_FILE.relative_to(D.ROOT))
