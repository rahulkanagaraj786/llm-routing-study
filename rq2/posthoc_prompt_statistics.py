"""Post hoc, prompts only: why does the frozen RQ2 router send coding questions to Flash?

  python rq2/posthoc_prompt_statistics.py logistic

Written after the RQ2 run (protocol log entry 24). It explains a result; it chooses nothing and
changes no result. It refits A's router exactly as the run did (same questions, same settings;
logistic regression is deterministic) and checks that the refit reproduces the gains on record
in results/rq2/<model>/predictions_rq2_transfer__*.csv. It then splits each set's mean predicted
gain into the part from the words and the part from the ten prompt statistics, by recomputing
the gains with one block held at its training average. Only prompt text is read for the target
sets: no score, cost or answer. Writes results/rq2/<model>/posthoc_prompt_statistics.json.
"""
import sys, csv, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from router import data as D, splits as S, registry, models as M

pos = [a for a in sys.argv[1:] if not a.startswith("--")]
if pos != ["logistic"]:
    sys.exit("only the logistic router has this structure (words + prompt statistics): python rq2/posthoc_prompt_statistics.py logistic")
family = pos[0]
BASE = D.ROOT / "results" / "rq2" / family
R = json.loads((BASE / "final_results_rq2.json").read_text())
config = R["config"]
assert config.get("use_stats"), "the router was fitted without the prompt statistics"

# refit A's router on the same development questions as the run (no unlock needed)
tr, _ = S.load_rows(f"posthoc:rq2:{family}:train", splits=("train",), benchmarks=tuple(D.SELECTION_POOL),
                    exclude=lambda r: not S.in_source_pool(r))
router = registry.make(family, config).fit(tr.prompts, tr.outcomes.y)
f, nt = router.features, router.features.n_terms

# prompts of each scored set, in the order of the predictions on record
by_qid = {r["qid"]: r["prompt"] for r in D.read_paired()}
names = [n for n in R["studies"]["rq2_transfer"]["targets"] if n != "hle_costs_repaired"]
S._log_access(f"posthoc:rq2:{family}:prompt_statistics", "prompt text only, no outcomes; " + ",".join(names),
              sum(R["studies"]["rq2_transfer"]["targets"][n]["n"] for n in names))


def gain(X):
    p = router.model.predict_proba(X)
    c = list(router.model.classes_)
    return p[:, c.index(1)] - p[:, c.index(-1)]


coef = router.model.coef_
cls = list(router.model.classes_)
w_gap = coef[cls.index(1)] - coef[cls.index(-1)]          # weight on the logit gap between "+1" and "-1"
out = dict(note=__doc__.split("\n\n")[1].replace("\n", " "), config=config, sets={},
           statistic_weights={n: float(w_gap[nt + i]) for i, n in enumerate(M.STAT_NAMES)},
           statistic_training_mean={n: float(f._mu[i]) for i, n in enumerate(M.STAT_NAMES)},
           statistic_training_sd={n: float(f._sd[i]) for i, n in enumerate(M.STAT_NAMES)})
for n in names:
    with open(BASE / f"predictions_rq2_transfer__{n}.csv") as fh:
        rec = list(csv.DictReader(fh))
    prompts = [by_qid[r["qid"]] for r in rec]
    X = f.transform(prompts)
    g = gain(X)
    g_rec = np.array([float(r["gain"]) for r in rec])
    assert np.allclose(g, g_rec, rtol=0, atol=1e-9), f"refit does not reproduce the recorded gains on {n}"
    stats_avg = X.tolil(copy=True); stats_avg[:, nt:] = 0       # statistics at their training average (z = 0)
    words_out = X.tolil(copy=True); words_out[:, :nt] = 0        # no word features at all
    z = X[:, nt:].toarray()
    out["sets"][n] = dict(n=len(prompts), mean_gain=float(g.mean()),
                          mean_gain_statistics_at_training_average=float(gain(stats_avg.tocsr()).mean()),
                          mean_gain_without_words=float(gain(words_out.tocsr()).mean()),
                          mean_z={s: float(v) for s, v in zip(M.STAT_NAMES, z.mean(axis=0))},
                          mean_contribution_to_logit_gap={s: float(v) for s, v in zip(M.STAT_NAMES, z.mean(axis=0) * w_gap[nt:])})
# what the router learned from: GPT-5's edge and prompt length by benchmark, training questions only
out["source_train_by_benchmark"] = {}
for b in sorted(set(tr.benchmark.tolist())):
    m = tr.benchmark == b
    o = tr.outcomes.take(np.flatnonzero(m))
    out["source_train_by_benchmark"][b] = dict(
        n=int(m.sum()), gpt5_accuracy=float(o.strong_correct.mean()), flash_accuracy=float(o.cheap_correct.mean()),
        advantage_pp=float(100 * (o.strong_correct.mean() - o.cheap_correct.mean())),
        median_words=float(np.median([len(p.split()) for p in tr.prompts[m]])),
        share_with_code_indicator=float(np.mean([bool(M._CODE.search(p)) for p in tr.prompts[m]])))
(BASE / "posthoc_prompt_statistics.json").write_text(json.dumps(out, indent=2))
for b, v in out["source_train_by_benchmark"].items():
    print(f"source train {b:9s} n={v['n']:5d}  GPT-5 edge {v['advantage_pp']:5.1f} pp  median words {v['median_words']:.0f}  "
          f"code indicator {100 * v['share_with_code_indicator']:.1f}%")
print(f"{'set':52s} {'mean gain':>9s} {'statistics at training average':>31s} {'without words':>14s}")
for n, v in out["sets"].items():
    print(f"{n:52s} {v['mean_gain']:9.3f} {v['mean_gain_statistics_at_training_average']:31.3f} {v['mean_gain_without_words']:14.3f}")
print("wrote", (BASE / "posthoc_prompt_statistics.json").relative_to(D.ROOT))
