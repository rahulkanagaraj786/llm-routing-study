"""The shared "fit, freeze the cutoffs, then score" step used by every final evaluation.

Both the RQ1 script (scripts/07_final_eval.py) and the RQ2 pipeline (rq2/) call run_one(),
so the two can never measure things differently.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from . import data as D, policy as P, registry


def list_price_outcomes(rows) -> D.Outcomes | None:
    """Outcomes with costs recomputed at the release's list prices (sensitivity check only)."""
    if any(r["strong_cost_list"] is None or r["cheap_cost_list"] is None for r in rows):
        return None
    return D.Outcomes(np.array([int(r["strong_score"]) for r in rows]), np.array([int(r["cheap_score"]) for r in rows]),
                      np.array([r["strong_cost_list"] for r in rows]), np.array([r["cheap_cost_list"] for r in rows]))


def report(gain, cutoffs, tab: D.Table, rows, label, out_dir: Path, manifest: dict, mix: tuple | None = None) -> dict:
    """Score frozen cutoffs on one table. `mix` = (benchmark, shares) adds the workload-mix check."""
    groups = np.array([manifest[q]["cluster_id"] for q in tab.qid])
    res = dict(label=label, n=len(tab), pp_per_question=100.0 / len(tab), allowances={},
               y_counts={str(v): int((tab.outcomes.y == v).sum()) for v in (-1, 0, 1)})
    lp = list_price_outcomes(rows)
    benches = set(tab.benchmark.tolist())
    for a, c in cutoffs.items():
        d = P.route_strong(gain, c["cutoff"])
        m = P.evaluate(d, tab.outcomes)
        m.update(P.evaluate_by_benchmark(d, tab.outcomes, tab.benchmark))
        m["bootstrap"] = P.bootstrap(d, tab.outcomes, groups=groups)
        m["within_benchmark_shuffle"] = P.within_benchmark_shuffle(d, tab.outcomes, tab.benchmark)
        if mix and mix[0] in benches and len(benches) > 1:
            m["workload_mix"] = [P.reweighted(d, tab.outcomes, tab.benchmark, mix[0], sh) for sh in mix[1]]
        m["cutoff"] = c["cutoff"]
        m["within_allowance"] = bool(m["accuracy_loss_pp"] <= float(a) + 1e-9)
        if lp is not None:
            ml = P.evaluate(d, lp)
            m["list_price_sensitivity"] = dict(gross_savings_pct=ml["gross_savings_pct"], cost=ml["cost"],
                                               always_strong_cost=ml["always_strong_cost"])
        res["allowances"][a] = m
    res["frontier"] = P.frontier(gain, tab.outcomes)
    np.savetxt(out_dir / f"predictions_{label}.csv",
               np.column_stack([tab.qid, tab.benchmark, gain] + [P.route_strong(gain, c["cutoff"]).astype(int) for c in cutoffs.values()]),
               fmt="%s", delimiter=",", header="qid,benchmark,gain," + ",".join(f"route_gpt5_at_{a}pp" for a in cutoffs), comments="")
    return res


def run_one(family: str, config: dict, label: str, tr: D.Table, va: D.Table, targets: dict, out_dir: Path, manifest: dict,
            mix_targets: tuple = (), mix: tuple | None = None) -> dict:
    """Fit on `tr`, choose the cutoffs on `va`, write them to disk, then score each target table.

    targets: name -> (Table, rows). The cutoff file is written before any target is scored."""
    print(f"[{label}] fit on {len(tr)} rows, calibrate on {len(va)} rows", flush=True)
    r = registry.make(family, config).fit(tr.prompts, tr.outcomes.y, va.prompts, va.outcomes.y)
    g_va = r.predict_gain(va.prompts)
    cutoffs = {f"{a:g}": P.select_cutoff(g_va, va.outcomes, a) for a in P.ALLOWANCES_PP}
    (out_dir / f"cutoffs_{label}.json").write_text(json.dumps(cutoffs, indent=2))
    r.save(out_dir / label / "model")
    cal = {a: P.evaluate(P.route_strong(g_va, c["cutoff"]), va.outcomes) for a, c in cutoffs.items()}
    out = dict(label=label, n_train=len(tr), n_calibration=len(va), fit_seconds=r.fit_seconds, cutoffs=cutoffs,
               calibration_metrics=cal, train_benchmarks=sorted(set(tr.benchmark.tolist())),
               calibration_benchmarks=sorted(set(va.benchmark.tolist())), targets={})
    for name, (tab, rows) in targets.items():
        t0 = time.perf_counter(); g = r.predict_gain(tab.prompts); dt = time.perf_counter() - t0
        res = report(g, cutoffs, tab, rows, f"{label}__{name}", out_dir, manifest, mix=mix if name in mix_targets else None)
        res["predict_ms_per_question"] = 1000 * dt / len(tab)
        out["targets"][name] = res
        m = res["allowances"]["1"]
        print(f"  {name}: n={len(tab)} | 1 pp: savings {m['gross_savings_pct']:.2f}% loss {m['accuracy_loss_pp']:.2f} pp "
              f"to-Flash {100*m['cheap_fraction']:.1f}%", flush=True)
    r.release()
    return out
