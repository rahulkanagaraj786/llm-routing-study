"""Development comparison for one model (docs/PROTOCOL.md, section 6).

  python scripts/05_dev_compare.py <model> --smoke   # 2-minute check on a small sample; writes nothing
  python scripts/05_dev_compare.py <model>           # the real run; resumable, one file per rotation

<model> is a name from src/router/registry.py, for example: logistic.

What it does, five times (once per rotation): learn on three folds, pick the best setting and
tune the cutoffs on a fourth, then score the fifth. Every development question ends up scored
by a router that never saw it. It reads only train+validation rows of the source pool
(SimpleQA, GPQA, MMLU-Pro without math and computer science). It never touches the test split.

Output: results/dev_selection/<model>/  (rotation_*.json, oof_predictions.csv, summary.json)
"""
import sys, json, time, csv
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from router import splits as S, policy as P, registry, runlog

args = [a for a in sys.argv[1:] if not a.startswith("--")]
if len(args) != 1:
    sys.exit(__doc__)
family = args[0]
registry.check(family)
SMOKE = "--smoke" in sys.argv
OUT = S.D.ROOT / "results" / "dev_selection" / family
log = lambda s: print(s, flush=True)

tab, folds = S.load_selection_dev(f"dev_compare:{family}{':smoke' if SMOKE else ''}")
if SMOKE:  # a small, fixed sample: enough to prove the model trains and predicts
    keep = np.arange(len(tab))[::8]
    tab, folds = tab.take(keep), folds[keep]
y = tab.outcomes.y
K = S.N_FOLDS
grid = registry.grid(family)
every_rotation = registry.FAMILIES[family]["tune_every_rotation"]


def fit_predict(config, fit, cal, ev):
    """Fit one setting; return its gains on the calibration and evaluation folds plus a record."""
    r = registry.make(family, config)
    r.fit(tab.prompts[fit], y[fit], tab.prompts[cal], y[cal])
    g_cal = r.predict_gain(tab.prompts[cal])
    t0 = time.perf_counter(); g_ev = r.predict_gain(tab.prompts[ev]); pred_s = time.perf_counter() - t0
    assert np.isfinite(g_ev).all() and (np.abs(g_ev) <= 1 + 1e-6).all(), "predict_gain must return finite values in [-1, 1]"
    info = dict(config=config, cal_advantage_mse=float(np.mean((g_cal - y[cal]) ** 2)), fit_seconds=r.fit_seconds,
                predict_seconds_per_question=pred_s / len(ev), **r.extra_info())
    r.release()
    return g_cal, g_ev, info


def rotation(k, configs):
    ev = np.flatnonzero(folds == k)
    cal = np.flatnonzero(folds == (k + 1) % K)
    fit = np.flatnonzero((folds != k) & (folds != (k + 1) % K))
    tried, best = [], None
    for c in configs:
        log(f"rotation {k}: fitting {family} {c} on {len(fit)} rows")
        g_cal, g_ev, info = fit_predict(c, fit, cal, ev)
        log(f"  cal_advantage_mse {info['cal_advantage_mse']:.5f}  fit {info['fit_seconds']:.1f}s")
        tried.append(info)
        if best is None or info["cal_advantage_mse"] < best[2]["cal_advantage_mse"] - 1e-12:
            best = (g_cal, g_ev, info)
    g_cal, g_ev, info = best
    cut = {f"{a:g}": P.select_cutoff(g_cal, tab.outcomes.take(cal), a) for a in P.ALLOWANCES_PP}
    return dict(rotation=k, eval_fold=k, calibration_fold=(k + 1) % K, n_fit=int(len(fit)), n_cal=int(len(cal)),
                n_eval=int(len(ev)), chosen=info, tried=tried, cutoffs=cut,
                eval_qid=tab.qid[ev].tolist(), eval_gain=[float(v) for v in g_ev])


if SMOKE:
    rot = rotation(0, grid[:1])
    g = np.array(rot["eval_gain"])
    ev = np.flatnonzero(folds == 0)
    m = P.evaluate(P.route_strong(g, rot["cutoffs"]["1"]["cutoff"]), tab.outcomes.take(ev))
    log(f"SMOKE OK: {family} trained on {rot['n_fit']} questions and scored {rot['n_eval']}; "
        f"gains from {g.min():.3f} to {g.max():.3f}; {100 * m['cheap_fraction']:.0f}% sent to Flash at the 1-point cutoff. "
        "Nothing was written. This is a wiring check, not a result.")
    sys.exit(0)

OUT.mkdir(parents=True, exist_ok=True)
rotations = []
frozen_config = None  # for models tuned in the first rotation only
for k in range(K):
    rot_path = OUT / f"rotation_{k}.json"
    if rot_path.exists():
        rot = json.loads(rot_path.read_text())
        log(f"rotation {k}: loaded from {rot_path.name}")
    else:
        rot = rotation(k, grid if (every_rotation or k == 0) else [frozen_config])
        rot_path.write_text(json.dumps(rot))
    rotations.append(rot)
    if not every_rotation and k == 0:
        frozen_config = rot["chosen"]["config"]

# ---- assemble out-of-fold predictions and decisions ----
pos = {q: i for i, q in enumerate(tab.qid)}
n = len(tab)
gain = np.full(n, np.nan); rot_of = np.full(n, -1)
decide = {f"{a:g}": np.zeros(n, dtype=bool) for a in P.ALLOWANCES_PP}
for rot in rotations:
    idx = np.array([pos[q] for q in rot["eval_qid"]])
    g = np.array(rot["eval_gain"])
    gain[idx] = g; rot_of[idx] = rot["rotation"]
    for a, c in rot["cutoffs"].items():
        decide[a][idx] = P.route_strong(g, c["cutoff"])
assert not np.isnan(gain).any() and (rot_of >= 0).all(), "saved rotations do not cover the current development questions"
with open(OUT / "oof_predictions.csv", "w", newline="") as f:
    w = csv.writer(f, lineterminator="\n")
    w.writerow(["qid", "benchmark", "rotation", "gain"] + [f"route_gpt5_at_{a}pp" for a in decide])
    for i in range(n):
        w.writerow([tab.qid[i], tab.benchmark[i], rot_of[i], repr(float(gain[i]))] + [int(decide[a][i]) for a in decide])

# the setting to freeze for the final fits: lowest mean calibration error across rotations
if every_rotation:
    by = {}
    for rot in rotations:
        for t in rot["tried"]:
            by.setdefault(json.dumps(t["config"], sort_keys=True), []).append(t["cal_advantage_mse"])
    mean_mse = {c: float(np.mean(v)) for c, v in by.items()}
    final_config = json.loads(min(mean_mse, key=mean_mse.get))
else:
    final_config, mean_mse = frozen_config, None
summary = dict(
    family=family, protocol_sha256=runlog.protocol_sha256(), env=runlog.env_info(rotations[0]["chosen"].get("device", "cpu")),
    n_questions=n, grid=grid, final_config=final_config, mean_cal_mse_by_config=mean_mse,
    chosen_config_per_rotation=[r["chosen"]["config"] for r in rotations],
    fit_seconds_per_rotation=[sum(t["fit_seconds"] for t in r["tried"]) for r in rotations],
    chosen_fit_seconds=[r["chosen"]["fit_seconds"] for r in rotations],
    predict_seconds_per_question=float(np.mean([r["chosen"]["predict_seconds_per_question"] for r in rotations])),
    cutoffs_per_rotation=[{a: c["cutoff"] for a, c in r["cutoffs"].items()} for r in rotations],
    mean_cal_advantage_mse_chosen=float(np.mean([r["chosen"]["cal_advantage_mse"] for r in rotations])),
)
(OUT / "summary.json").write_text(json.dumps(summary, indent=2))
log(f"wrote {OUT.relative_to(S.D.ROOT)}/oof_predictions.csv and summary.json | final config {final_config}")
