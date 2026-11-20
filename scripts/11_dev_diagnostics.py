"""Development diagnostics for every model with development results. Post hoc: nothing here
is used to choose settings, cutoffs or the primary model.

Reads only results/dev_selection/<model>/oof_predictions.csv and the development rows
(train + validation of all seven benchmarks, never the test split). Writes
results/dev_selection/diagnostics.json:

  per model     within-benchmark shuffle at each target, correlation of predicted gain with
                the outcome, share of the outcome's variance explained, the 1-point result in
                each of the five rotations, and savings read off the pooled curve at 0.5 to 2
                points of loss (descriptive and slightly optimistic: the cutoff is chosen on
                the same pooled predictions)
  ceiling       a router that knew the outcomes: GPT-5 only where only GPT-5 is right
  always_flash  for comparison
  headroom      why the savings are small: the outcome mix, each benchmark's share of the
                GPT-5 bill, how much knowing only the benchmark explains, and what each
                router sends to Flash at the 1-point target

  python scripts/11_dev_diagnostics.py
"""
import sys, csv, json
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from router import splits as S, policy as P, registry

BASE = S.D.ROOT / "results" / "dev_selection"
tab, _ = S.load_rq1_dev("dev_diagnostics")
out, bench = tab.outcomes, tab.benchmark
y = out.y
pos = {q: i for i, q in enumerate(tab.qid)}


def load_oof(family: str):
    rows = list(csv.DictReader(open(BASE / family / "oof_predictions.csv")))
    if len(rows) != len(tab) or {r["qid"] for r in rows} != set(tab.qid):
        sys.exit(f"{family}: out-of-fold predictions do not cover the development questions")
    g, rot = np.zeros(len(tab)), np.zeros(len(tab), int)
    dec = {a: np.zeros(len(tab), bool) for a in P.KEYS}   # True = GPT-5
    for r in rows:
        i = pos[r["qid"]]
        g[i], rot[i] = float(r["gain"]), int(r["rotation"])
        for a in dec:
            dec[a][i] = r[f"route_gpt5_at_{a}pp"] == "1"
    return g, rot, dec


res, sent_at_1pp = {}, {}
for fam in registry.names():
    if not (BASE / fam / "oof_predictions.csv").exists():
        continue
    g, rot, dec = load_oof(fam)
    matched = {}
    for a in (0.5, 1.0, 1.5, 2.0):
        m = P.evaluate(P.route_strong(g, P.select_cutoff(g, out, a)["cutoff"]), out)
        matched[str(a)] = dict(savings=round(m["gross_savings_pct"], 2), loss=round(m["accuracy_loss_pp"], 2))
    per = [P.evaluate(dec["1"][rot == k], out.take(rot == k)) for k in range(5)]
    res[fam] = dict(
        matched_loss=matched,
        corr_gain_y=float(np.corrcoef(g, y)[0, 1]),
        r2=float(1 - np.mean((g - y) ** 2) / np.var(y)),
        per_rotation_savings=[round(o["gross_savings_pct"], 1) for o in per],
        per_rotation_loss=[round(o["accuracy_loss_pp"], 2) for o in per],
        within_benchmark_shuffle={a: P.within_benchmark_shuffle(dec[a], out, bench) for a in P.KEYS},
    )
    sent_at_1pp[fam] = ~dec["1"]

orc = P.evaluate(y > 0, out)
res["ceiling"] = dict(savings=orc["gross_savings_pct"], loss=orc["accuracy_loss_pp"])
flash = P.evaluate(np.ones(len(tab), bool), out)
res["always_flash"] = dict(savings=flash["always_cheap_savings_pct"], loss=flash["always_cheap_loss_pp"])

# ---- headroom: why the savings are small ----
sc, cc = out.strong_cost, out.cheap_cost
sr, cr = out.strong_correct == 1, out.cheap_correct == 1
bill = sc.sum()
mix = lambda m: dict(both_right=float(np.mean(sr[m] & cr[m])), both_wrong=float(np.mean(~sr[m] & ~cr[m])),
                     only_gpt5_right=float(np.mean(y[m] == 1)), only_flash_right=float(np.mean(y[m] == -1)))
all_rows = np.ones(len(tab), bool)
per_bench = {}
for b in sorted(set(bench.tolist()), key=lambda b: -sc[bench == b].sum()):
    m = bench == b
    per_bench[b] = dict(n=int(m.sum()), share_of_questions=float(m.mean()), share_of_gpt5_bill=float(sc[m].sum() / bill),
                        gpt5_cost_per_question=float(sc[m].mean()), flash_cost_per_question=float(cc[m].mean()),
                        gpt5_accuracy=float(sr[m].mean()), flash_accuracy=float(cr[m].mean()), **mix(m))
bench_mean = {b: y[bench == b].mean() for b in per_bench}
mse_bench = float(np.mean((y - np.array([bench_mean[b] for b in bench])) ** 2))
sent = {}
for fam, s in sent_at_1pp.items():
    sent[fam] = dict(share_sent_to_flash=float(s.mean()), outcome_mix_of_sent=mix(s),
                     bill_saved_pct=float((sc[s] - cc[s]).sum() / bill * 100),
                     by_benchmark={b: dict(share_of_sent=float(np.mean(bench[s] == b)),
                                           bill_saved_pct=float((sc[s & (bench == b)] - cc[s & (bench == b)]).sum() / bill * 100))
                                   for b in per_bench})
res["headroom"] = dict(n=len(tab), outcome_mix=mix(all_rows), flash_bill_share_of_gpt5=float(cc.sum() / bill),
                       per_benchmark=per_bench, var_y=float(np.var(y)),
                       benchmark_only=dict(mse=mse_bench, variance_explained=1 - mse_bench / float(np.var(y)),
                                           note="each benchmark's average outcome as the prediction (in sample)"),
                       sent_to_flash_at_1pp=sent)
res["note"] = ("Post-hoc diagnostics on pooled out-of-fold gains and decisions. RQ1 development rows (all seven "
               "benchmarks). Not used for selection. Produced by scripts/11_dev_diagnostics.py.")
(BASE / "diagnostics.json").write_text(json.dumps(res, indent=1))

# ---- printout ----
for fam in sent_at_1pp:
    r = res[fam]; x = r["within_benchmark_shuffle"]["1"]
    print(f"{fam:9s} corr {r['corr_gain_y']:.3f}  variance explained {100 * r['r2']:.1f}%  "
          f"1 pp loss by rotation {r['per_rotation_loss']} (range {min(r['per_rotation_loss'])} to {max(r['per_rotation_loss'])})")
    print(f"          1 pp shuffle: router {x['router']['gross_savings_pct']:.1f}% / {x['router']['accuracy_loss_pp']:.2f} pp | "
          f"shuffled mean {x['shuffled_mean']['gross_savings_pct']:.1f}% / {x['shuffled_mean']['accuracy_loss_pp']:.2f} pp "
          f"(loss range {x['shuffled_range95']['accuracy_loss_pp'][0]:.2f} to {x['shuffled_range95']['accuracy_loss_pp'][1]:.2f}) | "
          f"shuffles at least as accurate {100 * x['share_of_shuffles_at_least_as_accurate']:.1f}%")
    print("          savings read off the curve at 0.5 / 1.0 / 1.5 / 2.0 pp: "
          + " / ".join(f"{v['savings']:.1f}%" for v in r["matched_loss"].values()))
h = res["headroom"]
print(f"ceiling {res['ceiling']['savings']:.1f}% saved, loss {res['ceiling']['loss']:.2f} pp | always Flash "
      f"{res['always_flash']['savings']:.1f}% saved, loss {res['always_flash']['loss']:.2f} pp")
print(f"outcome mix {({k: round(100 * v, 1) for k, v in h['outcome_mix'].items()})} | benchmark-only variance explained "
      f"{100 * h['benchmark_only']['variance_explained']:.1f}%")
print("wrote", (BASE / "diagnostics.json").relative_to(S.D.ROOT))
