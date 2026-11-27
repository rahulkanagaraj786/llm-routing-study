"""RQ2 workstream B: simulated few-shot adaptation on LiveCodeBench (PROTOCOL §8, log entry 25).

  python rq2/run_rq2_fewshot.py <model> --plan
  python rq2/run_rq2_fewshot.py <model> --dry-run        # source-pool rehearsal; no LCB test outcomes
  python rq2/run_rq2_fewshot.py <model> --unlock-final   # ONCE; refuses if results exist

k=0: A's saved predictions + A's original saved cutoffs (qid-filtered).
Retune-only: A's saved predictions; new cutoff on k labelled LCB-train questions.
Retrain: new fit on source train + k LCB-train; cutoffs on source-pool validation only.

Output: results/rq2/logistic/b_fewshot/ (dry-run: results/rq2_dryrun/logistic/b_fewshot/).
Does not modify A's files under results/rq2/logistic/.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from router import data as D, splits as S, policy as P, registry, runlog

DESIGN_SETTLED = True
K_GRID = (0, 10, 25, 50, 100, 200, 400)
N_SEEDS = 10
PRIMARY = "3"
ALPHA_BONF = 0.05 / len(K_GRID)
N_BOOT = 2000
BOOT_SEED = 221

args = sys.argv[1:]
pos = [a for a in args if not a.startswith("--")]
if len(pos) != 1:
    sys.exit(__doc__)
family = pos[0]
registry.check(family)
DRY = "--dry-run" in args
UNLOCK = "--unlock-final" in args
PLAN = "--plan" in args
if DRY + UNLOCK + PLAN != 1:
    sys.exit("choose exactly one of --plan, --dry-run or --unlock-final")
if not DESIGN_SETTLED:
    sys.exit("RQ2 B design not settled. See docs/PROTOCOL.md section 8.")

POOL = tuple(D.SELECTION_POOL)
A_DIR = D.ROOT / "results" / "rq2" / family
A_PRED = A_DIR / "predictions_rq2_transfer__code.csv"
A_CUT = A_DIR / "cutoffs_rq2_transfer.json"
AUDIT_DIR = A_DIR / "b_fewshot"
OUT = D.ROOT / "results" / ("rq2_dryrun" if DRY else "rq2") / family / "b_fewshot"
RESULT_FILE = OUT / "final_results_rq2_b.json"
dev_path = D.ROOT / "results" / "rq2_dev_selection" / family / "summary.json"


def load_json(path: Path):
    return json.loads(path.read_text())


def by_qid(tab: D.Table, rows: list[dict]) -> dict[str, int]:
    return {q: i for i, q in enumerate(tab.qid)}


def take_qids(tab: D.Table, rows: list[dict], qids: list[str]) -> tuple[D.Table, list[dict]]:
    ix = by_qid(tab, rows)
    idx = np.array([ix[q] for q in qids if q in ix], dtype=int)
    return tab.take(idx), [rows[i] for i in idx]


def concat_tables(a: D.Table, b: D.Table) -> D.Table:
    o = a.outcomes
    p = b.outcomes
    return D.Table(
        qid=np.concatenate([a.qid, b.qid]),
        benchmark=np.concatenate([a.benchmark, b.benchmark]),
        prompts=np.concatenate([a.prompts, b.prompts]),
        outcomes=D.Outcomes(
            np.concatenate([o.strong_correct, p.strong_correct]),
            np.concatenate([o.cheap_correct, p.cheap_correct]),
            np.concatenate([o.strong_cost, p.strong_cost]),
            np.concatenate([o.cheap_cost, p.cheap_cost]),
        ),
    )


def read_a_gains(path: Path) -> dict[str, float]:
    out = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            out[row["qid"]] = float(row["gain"])
    return out


def gains_for(tab: D.Table, gain_map: dict[str, float]) -> np.ndarray:
    missing = [q for q in tab.qid if q not in gain_map]
    if missing:
        sys.exit(f"missing gains for {len(missing)} qids (e.g. {missing[0]})")
    return np.array([gain_map[q] for q in tab.qid], dtype=float)


def score_cutoffs(gain: np.ndarray, cutoffs: dict, tab: D.Table, man: dict) -> dict:
    groups = np.array([man[q]["cluster_id"] for q in tab.qid])
    out = {}
    for a, c in cutoffs.items():
        d = P.route_strong(gain, float(c["cutoff"]))
        m = P.evaluate(d, tab.outcomes)
        m["cutoff"] = float(c["cutoff"])
        m["within_allowance"] = bool(m["accuracy_loss_pp"] <= float(a) + 1e-9)
        m["bootstrap"] = P.bootstrap(d, tab.outcomes, groups=groups)
        out[a] = m
    return out


def brief(m: dict) -> dict:
    return dict(
        gross_savings_pct=m["gross_savings_pct"],
        accuracy_loss_pp=m["accuracy_loss_pp"],
        cheap_fraction=m["cheap_fraction"],
        within_allowance=m["within_allowance"],
        gross_savings_pct_ci95=m["bootstrap"]["gross_savings_pct_ci95"],
        accuracy_loss_pp_ci95=m["bootstrap"]["accuracy_loss_pp_ci95"],
    )


def one_sided_t_upper(xs: list[float], alpha: float) -> float:
    x = np.asarray(xs, dtype=float)
    n = len(x)
    if n < 2:
        return float(x.mean()) if n else float("nan")
    mean = float(x.mean())
    se = float(x.std(ddof=1) / np.sqrt(n))
    if se == 0:
        return mean
    tcrit = float(stats.t.ppf(1 - alpha, df=n - 1))
    return mean + tcrit * se


def cluster_bootstrap_mean_loss(seed_decisions: list[np.ndarray], tab: D.Table, man: dict,
                                n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> list[float]:
    """Resample eval clusters; average loss across seed decision vectors each replicate."""
    groups = np.array([man[q]["cluster_id"] for q in tab.qid])
    _, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(inv.max() + 1)]
    G = len(members)
    rng = np.random.default_rng(seed)
    out_y = tab.outcomes
    means = []
    for _ in range(n_boot):
        pick = rng.integers(0, G, size=G)
        idx = np.concatenate([members[i] for i in pick])
        losses = []
        for dec in seed_decisions:
            sub = out_y.take(idx)
            d = dec[idx]
            m = P.evaluate(d, sub)
            losses.append(m["accuracy_loss_pp"])
        means.append(float(np.mean(losses)))
    return means


if PLAN:
    # Preflight only: reads audit JSON + A's saved prediction qids. No sealed LCB outcomes.
    lab = load_json(AUDIT_DIR / "label_pool_qids.json")
    ev = load_json(AUDIT_DIR / "eval_qids.json")
    if not A_PRED.exists() or not A_CUT.exists():
        sys.exit("missing A's saved predictions or cutoffs")
    a_qids = set()
    with A_PRED.open() as f:
        next(f)
        for line in f:
            a_qids.add(line.split(",", 1)[0])
    missing = sorted(set(ev["qids"]) - a_qids)
    print("RQ2 workstream B preflight (few-shot adaptation on LiveCodeBench)")
    print(f"  label pool (LCB train): {lab['n']}  (frozen; expect 633)")
    print(f"  eval (LCB test):        {ev['n']}  (frozen; expect 211)")
    print(f"  LCB val unused:         211")
    print(f"  k grid (full, no shrink): {list(K_GRID)}  seeds: {N_SEEDS}")
    print(f"  max k vs label pool:    {max(K_GRID)} <= {lab['n']}: {max(K_GRID) <= lab['n']}")
    print(f"  A predictions: {A_PRED.relative_to(D.ROOT)} (n={len(a_qids)})")
    print(f"  A cutoffs:     {A_CUT.relative_to(D.ROOT)}")
    print(f"  A covers eval qids: {len(missing) == 0}" + (f"  missing={missing[:3]}..." if missing else ""))
    print(f"  sealed LCB outcomes read: no (--plan only)")
    print(f"  output:        {OUT.relative_to(D.ROOT)}")
    if lab["n"] != 633 or ev["n"] != 211:
        sys.exit("frozen qid counts differ from the locked design (633 / 211)")
    if max(K_GRID) > lab["n"]:
        sys.exit("k grid exceeds label pool")
    if missing:
        sys.exit("A's saved predictions do not cover the frozen eval set")
    sys.exit(0)

if UNLOCK and RESULT_FILE.exists():
    sys.exit(f"{RESULT_FILE.relative_to(D.ROOT)} already exists: B unlock runs once. Do not delete to rerun.")
if not dev_path.exists():
    sys.exit(f"missing {dev_path.relative_to(D.ROOT)}: run scripts/05_dev_compare.py {family} --source-pool")
if not A_PRED.exists() or not A_CUT.exists():
    sys.exit("missing A's RQ2 transfer predictions or cutoffs")
if not (AUDIT_DIR / "label_pool_qids.json").exists():
    sys.exit("run rq2/audit_rq2_b_overlap.py first")

config = load_json(dev_path)["final_config"]
man = S.read_manifest()
print(f"model: {family}  study: RQ2-B  settings: {config}  mode: {'DRY RUN' if DRY else 'FINAL'}", flush=True)
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------------------------
# Questions
# ---------------------------------------------------------------------------------------------
if DRY:
    # Source-pool rehearsal only. GPQA stands in for coding; no holdout/LCB loads.
    tab, folds = S.load_selection_dev(f"rq2_b_dry_run:{family}")
    rows_by = {r["qid"]: r for r in D.read_paired() if r["qid"] in set(tab.qid)}
    rows = [rows_by[q] for q in tab.qid]
    fold = {q: int(f) for q, f in zip(tab.qid, folds)}
    tr_tab, tr_rows = take_qids(tab, rows, [q for q in tab.qid if fold[q] <= 2 and rows_by[q]["benchmark"] != "gpqa"])
    va_tab, va_rows = take_qids(tab, rows, [q for q in tab.qid if fold[q] == 3 and rows_by[q]["benchmark"] != "gpqa"])
    # GPQA is smaller than LCB; use all of it, 70/30 label/eval, and shrink the k grid.
    code_q = [q for q in tab.qid if rows_by[q]["benchmark"] == "gpqa"]
    rng = np.random.default_rng(221)
    order = rng.permutation(code_q).tolist()
    n_lab = max(40, int(0.7 * len(order)))
    label_pool = order[:n_lab]
    eval_qids = order[n_lab:]
    if len(eval_qids) < 15 or len(label_pool) < 25:
        sys.exit(f"dry-run GPQA split too small: label={len(label_pool)} eval={len(eval_qids)}")
    # Stand-in "A": fit on source train (no GPQA), cutoffs on source val.
    a_router = registry.make(family, config).fit(tr_tab.prompts, tr_tab.outcomes.y, va_tab.prompts, va_tab.outcomes.y)
    a_cutoffs = {f"{a:g}": P.select_cutoff(a_router.predict_gain(va_tab.prompts), va_tab.outcomes, a)
                 for a in P.ALLOWANCES_PP}
    code_tab, code_rows = take_qids(tab, rows, label_pool + eval_qids)
    a_gain_map = {q: float(g) for q, g in zip(code_tab.qid, a_router.predict_gain(code_tab.prompts))}
    a_router.release()
    lab_tab, lab_rows = take_qids(tab, rows, label_pool)
    ev_tab, ev_rows = take_qids(tab, rows, eval_qids)
    k_grid = [k for k in K_GRID if k == 0 or k <= len(label_pool)]
else:
    label_pool = load_json(AUDIT_DIR / "label_pool_qids.json")["qids"]
    eval_qids = load_json(AUDIT_DIR / "eval_qids.json")["qids"]
    tag = f"final:rq2_b:{family}"
    tr_tab, tr_rows = S.load_rows(f"{tag}:source_train", splits=("train",), benchmarks=POOL,
                                  exclude=lambda r: not S.in_source_pool(r))
    va_tab, va_rows = S.load_rows(f"{tag}:source_val", splits=("val",), benchmarks=POOL,
                                  exclude=lambda r: not S.in_source_pool(r))
    lcb_tr = S.load_rows(f"{tag}:lcb_train", splits=("train",), benchmarks=("livecodebench",), unlock_final=True)
    lcb_te = S.load_rows(f"{tag}:lcb_test", splits=("test",), benchmarks=("livecodebench",), unlock_final=True)
    lab_tab, lab_rows = take_qids(lcb_tr[0], lcb_tr[1], label_pool)
    ev_tab, ev_rows = take_qids(lcb_te[0], lcb_te[1], eval_qids)
    assert len(lab_tab) == len(label_pool) and len(ev_tab) == len(eval_qids)
    a_gain_map = read_a_gains(A_PRED)
    a_cutoffs = load_json(A_CUT)
    k_grid = list(K_GRID)

# ---------------------------------------------------------------------------------------------
# Run arms
# ---------------------------------------------------------------------------------------------
results = dict(
    mode="dry_run" if DRY else "final",
    study="rq2_b",
    family=family,
    label=registry.label(family),
    config=config,
    protocol_sha256=runlog.protocol_sha256(),
    env=runlog.env_info(),
    design=dict(
        protocol="docs/PROTOCOL.md section 8, log entry 25",
        rehearsal=DRY,
        k_grid=list(k_grid),
        n_seeds=N_SEEDS,
        primary_allowance_pp=float(PRIMARY),
        label_pool_n=len(label_pool),
        eval_n=len(ev_tab),
    ),
    arms={},
)

# k=0: A's predictions + A's original cutoffs on eval
g_eval_a = gains_for(ev_tab, a_gain_map)
m0 = score_cutoffs(g_eval_a, a_cutoffs, ev_tab, man)
results["arms"]["k0"] = dict(
    description="A saved predictions + A original saved cutoffs on B eval",
    n_eval=len(ev_tab),
    allowances={a: brief(m) for a, m in m0.items()},
)
print(f"[k=0] eval n={len(ev_tab)} | 3 pp: savings {m0[PRIMARY]['gross_savings_pct']:.2f}% "
      f"loss {m0[PRIMARY]['accuracy_loss_pp']:.2f} pp", flush=True)

# Nested seeds for k>0
retune_curves = {a: {k: [] for k in k_grid if k > 0} for a in P.KEYS}
retrain_curves = {a: {k: [] for k in k_grid if k > 0} for a in P.KEYS}
retune_decisions_primary = {k: [] for k in k_grid if k > 0}
retrain_decisions_primary = {k: [] for k in k_grid if k > 0}
seed_orders = {}

g_lab_a = gains_for(lab_tab, a_gain_map)

for seed in range(N_SEEDS):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(label_pool)).tolist()
    seed_orders[str(seed)] = [label_pool[i] for i in order]
    print(f"[seed {seed}] nested prefixes…", flush=True)
    for k in k_grid:
        if k == 0:
            continue
        pref_idx = order[:k]
        pref_q = [label_pool[i] for i in pref_idx]
        # indices into lab_tab (lab_tab is ordered by label_pool)
        lab_ix = by_qid(lab_tab, lab_rows)
        idx = np.array([lab_ix[q] for q in pref_q], dtype=int)
        g_k = g_lab_a[idx]
        out_k = lab_tab.outcomes.take(idx)
        # Retune-only: new cutoffs on k labels, A's gains on eval
        cut_retune = {f"{a:g}": P.select_cutoff(g_k, out_k, a) for a in P.ALLOWANCES_PP}
        m_ret = score_cutoffs(g_eval_a, cut_retune, ev_tab, man)
        for a in P.KEYS:
            retune_curves[a][k].append(brief(m_ret[a]))
        retune_decisions_primary[k].append(P.route_strong(g_eval_a, float(cut_retune[PRIMARY]["cutoff"])))

        # Retrain: source train + k labelled coding rows; cutoffs on source val
        k_tab, k_rows = take_qids(lab_tab, lab_rows, pref_q)
        fit_tab = concat_tables(tr_tab, k_tab)
        router = registry.make(family, config).fit(
            fit_tab.prompts, fit_tab.outcomes.y, va_tab.prompts, va_tab.outcomes.y)
        g_va = router.predict_gain(va_tab.prompts)
        cut_retrain = {f"{a:g}": P.select_cutoff(g_va, va_tab.outcomes, a) for a in P.ALLOWANCES_PP}
        g_ev = router.predict_gain(ev_tab.prompts)
        m_rt = score_cutoffs(g_ev, cut_retrain, ev_tab, man)
        for a in P.KEYS:
            retrain_curves[a][k].append(brief(m_rt[a]))
        retrain_decisions_primary[k].append(P.route_strong(g_ev, float(cut_retrain[PRIMARY]["cutoff"])))
        router.release()

# Aggregate curves + Bonferroni success on retrain @ 3 pp
def aggregate(curves_for_allowance: dict) -> dict:
    out = {}
    for k, runs in curves_for_allowance.items():
        losses = [r["accuracy_loss_pp"] for r in runs]
        savs = [r["gross_savings_pct"] for r in runs]
        out[str(k)] = dict(
            n_seeds=len(runs),
            mean_accuracy_loss_pp=float(np.mean(losses)),
            mean_gross_savings_pct=float(np.mean(savs)),
            mean_cheap_fraction=float(np.mean([r["cheap_fraction"] for r in runs])),
            seed_accuracy_loss_pp=losses,
            seed_gross_savings_pct=savs,
            t_upper_loss_bonferroni=one_sided_t_upper(losses, ALPHA_BONF),
            t_interval_loss_95=[float(np.mean(losses) - stats.t.ppf(0.975, len(losses) - 1) * np.std(losses, ddof=1) / np.sqrt(len(losses))),
                                float(np.mean(losses) + stats.t.ppf(0.975, len(losses) - 1) * np.std(losses, ddof=1) / np.sqrt(len(losses)))] if len(losses) > 1 else [float(losses[0]), float(losses[0])],
        )
    return out


retune_agg = {a: aggregate(retune_curves[a]) for a in P.KEYS}
retrain_agg = {a: aggregate(retrain_curves[a]) for a in P.KEYS}

selected_k = None
for k in k_grid:
    if k == 0:
        continue
    up = retrain_agg[PRIMARY][str(k)]["t_upper_loss_bonferroni"]
    if up <= float(PRIMARY) + 1e-9:
        selected_k = k
        break

success = dict(
    rule="smallest k with Bonferroni one-sided t upper bound on mean retrain loss <= 3 pp",
    alpha_individual=ALPHA_BONF,
    m_comparisons=len(K_GRID),
    selected_k=selected_k,
    message=("not within tested range" if selected_k is None else f"selected k={selected_k}"),
)

# Cluster bootstrap for k=0 (single decision) and selected k (ten-seed mean)
k0_dec = P.route_strong(g_eval_a, float(a_cutoffs[PRIMARY]["cutoff"]))
boot = {}
boot_k0 = cluster_bootstrap_mean_loss([k0_dec], ev_tab, man)
boot["k0"] = dict(
    mean_loss_pp=float(np.mean(boot_k0)),
    ci95=[float(np.quantile(boot_k0, 0.025)), float(np.quantile(boot_k0, 0.975))],
    note="cluster bootstrap of loss for A's frozen decisions on B eval (one decision vector)",
)
if selected_k is not None:
    boot_sel = cluster_bootstrap_mean_loss(retrain_decisions_primary[selected_k], ev_tab, man)
    boot["retrain_selected_k"] = dict(
        k=selected_k,
        mean_loss_pp=float(np.mean(boot_sel)),
        ci95=[float(np.quantile(boot_sel, 0.025)), float(np.quantile(boot_sel, 0.975))],
        note="cluster bootstrap of ten-seed mean retrain loss at selected k",
    )

results["arms"]["retune_only"] = dict(
    description="A saved predictions; new cutoff on k LCB-train labels",
    primary_allowance=PRIMARY,
    by_allowance=retune_agg,
)
results["arms"]["retrain"] = dict(
    description="new B fit on source train + k LCB-train; cutoff on source val",
    primary_allowance=PRIMARY,
    by_allowance=retrain_agg,
)
results["success"] = success
results["cluster_bootstrap_primary"] = boot
results["seed_label_orders"] = seed_orders
if not DRY:
    results["audit"] = dict(
        overlap_audit_sha256=D.sha256_file(AUDIT_DIR / "overlap_audit.json"),
        label_pool_sha256=D.sha256_file(AUDIT_DIR / "label_pool_qids.json"),
        eval_sha256=D.sha256_file(AUDIT_DIR / "eval_qids.json"),
        a_predictions_sha256=D.sha256_file(A_PRED),
        a_cutoffs_sha256=D.sha256_file(A_CUT),
    )
    results["test_reads"] = dict(
        note="sealed LCB train+test reads for labelling and eval; A's artifacts were only read",
    )

RESULT_FILE.write_text(json.dumps(results, indent=2) + "\n")
(OUT / "cutoffs_rq2_b_k0_from_A.json").write_text(json.dumps(a_cutoffs, indent=2) + "\n")
print(f"wrote {RESULT_FILE.relative_to(D.ROOT)}")
print(f"success: {success['message']}")
if selected_k is not None:
    print(f"  retrain k={selected_k}: mean loss "
          f"{retrain_agg[PRIMARY][str(selected_k)]['mean_accuracy_loss_pp']:.2f} pp, "
          f"Bonferroni upper {retrain_agg[PRIMARY][str(selected_k)]['t_upper_loss_bonferroni']:.2f}")
