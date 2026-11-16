"""Nested 5-rotation development comparison on the selection pool (docs/PROTOCOL.md, section 6).

  python scripts/05_dev_compare.py logistic
  python scripts/05_dev_compare.py forest
  python scripts/05_dev_compare.py modernbert      # resumable; one JSON per rotation

Reads only train+val rows of the source pool: SimpleQA, GPQA and MMLU-Pro without its
math and computer-science questions."""
import sys, json, time, csv
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from router import splits as S, policy as P, models as M, runlog

family = sys.argv[1]
OUT = S.D.ROOT / "results" / "dev_selection" / family
OUT.mkdir(parents=True, exist_ok=True)
tab, folds = S.load_selection_dev(f"dev_compare:{family}")
y = tab.outcomes.y
K = S.N_FOLDS
log = lambda s: print(s, flush=True)


def make(config):
    if family == "modernbert":
        from router import modernbert as MB
        base = MB.REMOTE_GPU if MB.pick_device().type == "cuda" else MB.LOCAL_MPS
        return MB.ModernBertRouter(**{**base, **config})
    return M.make(family, config)


def fit_predict(config, fit, cal, ev):
    r = make(config)
    if family == "modernbert":
        r.fit(tab.prompts[fit], y[fit], tab.prompts[cal], y[cal], log=log)
    else:
        r.fit(tab.prompts[fit], y[fit])
    g_cal = r.predict_gain(tab.prompts[cal])
    t0 = time.perf_counter(); g_ev = r.predict_gain(tab.prompts[ev]); pred_s = time.perf_counter() - t0
    info = dict(config=config, cal_advantage_mse=float(np.mean((g_cal - y[cal]) ** 2)), fit_seconds=r.fit_seconds,
                predict_seconds_per_question=pred_s / len(ev))
    if family == "modernbert":
        info.update(history=r.history, best_epoch=r.best_epoch, device=str(r.device),
                    truncation_rate_fit=r.train_truncation_rate, full_config=r.config)
        del r.model
        import torch, gc
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    elif family == "logistic":
        info.update(n_iter=r.n_iter, n_terms=r.features.n_terms)
    return g_cal, g_ev, info


if family == "logistic":
    grid = M.LOGISTIC_GRID
elif family == "forest":
    grid = M.FOREST_GRID
else:
    from router import modernbert as MB
    grid = MB.MODERNBERT_GRID

rotations = []
frozen_config = None  # ModernBERT: chosen in rotation 0, reused afterwards
for k in range(K):
    rot_path = OUT / f"rotation_{k}.json"
    if rot_path.exists():
        rot = json.loads(rot_path.read_text())
        log(f"rotation {k}: loaded from {rot_path.name}")
    else:
        ev = np.flatnonzero(folds == k)
        cal = np.flatnonzero(folds == (k + 1) % K)
        fit = np.flatnonzero((folds != k) & (folds != (k + 1) % K))
        configs = grid if (family != "modernbert" or k == 0) else [frozen_config]
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
        rot = dict(rotation=k, eval_fold=k, calibration_fold=(k + 1) % K, n_fit=int(len(fit)), n_cal=int(len(cal)),
                   n_eval=int(len(ev)), chosen=info, tried=tried, cutoffs=cut,
                   eval_qid=tab.qid[ev].tolist(), eval_gain=[float(v) for v in g_ev])
        rot_path.write_text(json.dumps(rot))
    rotations.append(rot)
    if family == "modernbert" and k == 0:
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
assert not np.isnan(gain).any() and (rot_of >= 0).all()
with open(OUT / "oof_predictions.csv", "w", newline="") as f:
    w = csv.writer(f, lineterminator="\n")
    w.writerow(["qid", "benchmark", "rotation", "gain"] + [f"route_gpt5_at_{a}pp" for a in decide])
    for i in range(n):
        w.writerow([tab.qid[i], tab.benchmark[i], rot_of[i], repr(float(gain[i]))] + [int(decide[a][i]) for a in decide])

# configuration to freeze for later fits: lowest mean calibration MSE across rotations
if family == "modernbert":
    final_config = frozen_config
    mean_mse = None
else:
    by = {}
    for rot in rotations:
        for t in rot["tried"]:
            by.setdefault(json.dumps(t["config"], sort_keys=True), []).append(t["cal_advantage_mse"])
    mean_mse = {c: float(np.mean(v)) for c, v in by.items()}
    final_config = json.loads(min(mean_mse, key=mean_mse.get))
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
