"""Apply the pre-specified selection rule (docs/PROTOCOL.md, section 6) to the pooled
out-of-fold decisions of each candidate.

  python scripts/06_select_algorithm.py                 # needs the development results of every model in the registry
  python scripts/06_select_algorithm.py --defer gbm     # provisional: compare the others while one is still pending
  python scripts/06_select_algorithm.py --drop encoder  # team decision that a model will not be run
  python scripts/06_select_algorithm.py --partial           # descriptive table for those available; no winner
"""
import sys, json, csv
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from router import splits as S, policy as P, runlog, registry

ORDER = registry.names()  # simplest router first
QUALIFY_TOLERANCE_PP = 0.5
MIN_IMPROVEMENT_PP = 1.0  # a more complex router must save at least this much more (protocol section 6)
partial = "--partial" in sys.argv
DEFERRED = [sys.argv[i + 1] for i, a in enumerate(sys.argv) if a == "--defer"]
DROPPED = [sys.argv[i + 1] for i, a in enumerate(sys.argv) if a == "--drop"]
assert all(d in ORDER for d in DEFERRED + DROPPED), (DEFERRED, DROPPED)
REQUIRED = [f for f in ORDER if f not in DEFERRED + DROPPED]
BASE = S.D.ROOT / "results" / "dev_selection"

tab, folds = S.load_selection_dev("select_algorithm")
man = S.read_manifest()
groups = np.array([man[q]["cluster_id"] for q in tab.qid])
pos = {q: i for i, q in enumerate(tab.qid)}
out = tab.outcomes

avail, decisions, gains, summaries = [], {}, {}, {}
for fam in REQUIRED:
    f = BASE / fam / "oof_predictions.csv"
    if not f.exists():
        continue
    rows = list(csv.DictReader(open(f)))
    assert len(rows) == len(tab) and {r["qid"] for r in rows} == set(pos), f"{fam}: predictions do not cover the selection pool"
    d = {a: np.zeros(len(tab), dtype=bool) for a in ("0", "1", "2")}
    g = np.zeros(len(tab))
    for r in rows:
        i = pos[r["qid"]]
        g[i] = float(r["gain"])
        for a in d:
            d[a][i] = r[f"route_gpt5_at_{a}pp"] == "1"
    decisions[fam], gains[fam] = d, g
    summaries[fam] = json.loads((BASE / fam / "summary.json").read_text())
    avail.append(fam)
if not partial and avail != REQUIRED:
    sys.exit(f"only {avail} available; run with --partial for a descriptive table (no winner is declared)")

res = dict(protocol_sha256=runlog.protocol_sha256(), env=runlog.env_info(), n_questions=len(tab), candidates={},
           deferred_candidates=DEFERRED, dropped_candidates=DROPPED,
           candidate_protocol_hashes={f: summaries[f]["protocol_sha256"] for f in avail})
base = P.evaluate(np.ones(len(tab), dtype=bool), out)
res["baselines"] = dict(always_gpt5=dict(accuracy=base["always_strong_accuracy"], cost=base["always_strong_cost"]),
                        always_flash=dict(accuracy=base["always_cheap_accuracy"], cost=base["always_cheap_cost"],
                                          gross_savings_pct=base["always_cheap_savings_pct"], accuracy_loss_pp=base["always_cheap_loss_pp"]),
                        y_counts={str(v): int((out.y == v).sum()) for v in (-1, 0, 1)})
for fam in avail:
    c = dict(final_config=summaries[fam]["final_config"], predict_ms_per_question=1000 * summaries[fam]["predict_seconds_per_question"],
             total_fit_seconds=float(sum(summaries[fam]["fit_seconds_per_rotation"])),
             mean_cal_advantage_mse=summaries[fam]["mean_cal_advantage_mse_chosen"],
             oof_advantage_mse=float(np.mean((gains[fam] - out.y) ** 2)), device=summaries[fam]["env"]["device"], allowances={})
    for a in ("0", "1", "2"):
        d = decisions[fam][a]
        m = P.evaluate(d, out)
        m.update(P.evaluate_by_benchmark(d, out, tab.benchmark))
        m["bootstrap"] = P.bootstrap(d, out, groups=groups)
        c["allowances"][a] = m
    res["candidates"][fam] = c

# pairwise paired-bootstrap differences at the primary allowance
a = f"{P.PRIMARY_ALLOWANCE_PP:g}"
res["pairwise_at_primary"] = {}
for i, hi in enumerate(avail):
    for lo in avail[:i]:
        b = P.bootstrap(decisions[hi][a], out, groups=groups, other=decisions[lo][a])
        res["pairwise_at_primary"][f"{hi}_minus_{lo}"] = {k: b[k] for k in ("diff_savings_pct_ci90", "diff_savings_pct_ci95", "diff_loss_pp_ci95", "prob_savings_higher")}

# ---- selection rule ----
if not partial:
    S1 = {f: res["candidates"][f]["allowances"][a]["gross_savings_pct"] for f in REQUIRED}
    L1 = {f: res["candidates"][f]["allowances"][a]["accuracy_loss_pp"] for f in REQUIRED}
    qualified = [f for f in REQUIRED if L1[f] <= P.PRIMARY_ALLOWANCE_PP + QUALIFY_TOLERANCE_PP + 1e-9]
    steps = []
    if not qualified:
        winner = min(REQUIRED, key=lambda f: L1[f]); strict = winner
        steps.append("no candidate qualified; selected the smallest out-of-fold loss")
    else:
        strict = max(qualified, key=lambda f: S1[f])
        winner = qualified[0]
        for f in REQUIRED:
            if f not in qualified:
                steps.append(f"{f} does not qualify: out-of-fold loss {L1[f]:.2f} pp is above the {P.PRIMARY_ALLOWANCE_PP + QUALIFY_TOLERANCE_PP:g} pp limit")
        steps.append(f"start with the simplest qualified router: {winner}")
        for ch in qualified[1:]:
            lo90 = res["pairwise_at_primary"][f"{ch}_minus_{winner}"]["diff_savings_pct_ci90"][0]
            gain = S1[ch] - S1[winner]
            if gain >= MIN_IMPROVEMENT_PP and lo90 > 0:
                steps.append(f"{ch} replaces {winner}: saves {gain:.2f} points more (minimum {MIN_IMPROVEMENT_PP}) and the 90% interval lower bound is {lo90:.2f} > 0")
                winner = ch
            else:
                steps.append(f"{ch} does not replace {winner}: savings difference {gain:.2f} points (minimum {MIN_IMPROVEMENT_PP}), 90% interval lower bound {lo90:.2f}")
    res["selection"] = dict(primary_allowance_pp=P.PRIMARY_ALLOWANCE_PP, qualify_tolerance_pp=QUALIFY_TOLERANCE_PP, min_improvement_pp=MIN_IMPROVEMENT_PP, savings=S1, loss=L1,
                            qualified=qualified, strict_highest_savings=strict, selected=winner, steps=steps,
                            rules_agree=strict == winner, selected_config=res["candidates"][winner]["final_config"])
name = "selection_partial.json" if partial else "selection.json"
(BASE / name).write_text(json.dumps(res, indent=2))

# ---- console report ----
b = res["baselines"]
print(f"Source pool, development rows: n = {len(tab)}   (protocol {res['protocol_sha256'][:12]})")
print(f"always GPT-5: accuracy {100*b['always_gpt5']['accuracy']:.2f}%  cost ${b['always_gpt5']['cost']:.2f}")
print(f"always Flash: accuracy {100*b['always_flash']['accuracy']:.2f}%  cost ${b['always_flash']['cost']:.2f}  savings {b['always_flash']['gross_savings_pct']:.1f}%  loss {b['always_flash']['accuracy_loss_pp']:.2f} pp")
print(f"\n{'candidate':11s} {'allow':>5s} {'savings %':>9s} {'95% CI':>15s} {'loss pp':>8s} {'95% CI':>15s} {'to Flash':>8s} {'macro sav':>9s} {'macro loss':>10s}")
for fam in avail:
    for al in ("0", "1", "2"):
        m = res["candidates"][fam]["allowances"][al]; bs = m["bootstrap"]; eq = m["equal_benchmark_average"]
        print(f"{fam:11s} {al+' pp':>5s} {m['gross_savings_pct']:9.2f} [{bs['gross_savings_pct_ci95'][0]:5.2f},{bs['gross_savings_pct_ci95'][1]:6.2f}] {m['accuracy_loss_pp']:8.2f} "
              f"[{bs['accuracy_loss_pp_ci95'][0]:5.2f},{bs['accuracy_loss_pp_ci95'][1]:6.2f}] {100*m['cheap_fraction']:7.1f}% {eq['gross_savings_pct']:9.2f} {eq['accuracy_loss_pp']:10.2f}")
print(f"\nper benchmark at {a} pp (savings % / loss pp / share to Flash):")
for fam in avail:
    per = res["candidates"][fam]["allowances"][a]["per_benchmark"]
    print(f"  {fam:11s} " + "   ".join(f"{bn}: {v['gross_savings_pct']:.1f} / {v['accuracy_loss_pp']:.2f} / {100*v['cheap_fraction']:.0f}%" for bn, v in per.items()))
print("\noverhead:")
for fam in avail:
    c = res["candidates"][fam]
    print(f"  {fam:11s} predict {c['predict_ms_per_question']:.3f} ms/question on {c['device']}; all fits {c['total_fit_seconds']:.0f}s; OOF advantage MSE {c['oof_advantage_mse']:.4f}; config {c['final_config']}")
print("\npairwise savings difference at the primary allowance (paired bootstrap):")
for k, v in res["pairwise_at_primary"].items():
    print(f"  {k}: 90% CI [{v['diff_savings_pct_ci90'][0]:.2f}, {v['diff_savings_pct_ci90'][1]:.2f}]  P(higher) {v['prob_savings_higher']:.2f}")
if not partial:
    s = res["selection"]
    print("\nSELECTION:", s["selected"], "| highest savings among qualified:", s["strict_highest_savings"], "| qualified:", s["qualified"],
          "| deferred:", DEFERRED or "none", "| dropped:", DROPPED or "none")
    for st in s["steps"]:
        print("  -", st)
else:
    print("\nPARTIAL: no winner declared until all three candidates are available.")
