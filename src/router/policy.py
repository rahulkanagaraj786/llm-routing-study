"""Threshold routing policy: cutoff selection, fixed-policy evaluation, bootstrap.

Policy family (agreed by the team): choose GPT-5 when gain > cutoff, otherwise Flash.
The cutoff is chosen on calibration/validation data by minimising recorded cost
subject to accuracy >= always-GPT-5 accuracy - allowance. It is then frozen.
"""
from __future__ import annotations

import numpy as np

from .data import Outcomes

ALLOWANCES_PP = (0.0, 1.0, 2.0, 3.0, 5.0)
# 0, 1 and 2 were fixed before any test. 3 and 5 were added on 2026-11-16, after logistic
# regression's first test result was known (docs/PROTOCOL.md, log entry 12).
PRE_DECLARED_PP = (0.0, 1.0, 2.0)
KEYS = tuple(f"{a:g}" for a in ALLOWANCES_PP)            # ("0", "1", "2", "3", "5")
ADDED_LATER_KEYS = tuple(f"{a:g}" for a in ALLOWANCES_PP if a not in PRE_DECLARED_PP)
PRIMARY_ALLOWANCE_PP = 1.0
_EPS = 1e-12


def candidate_cutoffs(gain: np.ndarray) -> np.ndarray:
    """-inf (always GPT-5), midpoints between adjacent distinct gains, +inf (always Flash).

    Midpoints give the same decisions on the calibration data as cutting at an observed
    value, but are not sensitive to last-digit floating-point differences across machines."""
    u = np.unique(gain)
    mids = (u[:-1] + u[1:]) / 2.0
    return np.concatenate(([-np.inf], mids, [np.inf]))


def route_strong(gain: np.ndarray, cutoff: float) -> np.ndarray:
    return gain > cutoff


def select_cutoff(gain: np.ndarray, out: Outcomes, allowance_pp: float) -> dict:
    """Minimise recorded cost subject to the accuracy-loss allowance.
    Ties: higher accuracy, then the higher cutoff (deterministic)."""
    n = len(gain)
    order = np.argsort(gain, kind="stable")
    g = gain[order]
    sc, cc = out.strong_correct[order].astype(float), out.cheap_correct[order].astype(float)
    sk, ck = out.strong_cost[order], out.cheap_cost[order]
    # k = number of lowest-gain questions sent to Flash
    corr_k = np.concatenate(([0.0], np.cumsum(cc))) + (sc.sum() - np.concatenate(([0.0], np.cumsum(sc))))
    cost_k = np.concatenate(([0.0], np.cumsum(ck))) + (sk.sum() - np.concatenate(([0.0], np.cumsum(sk))))
    # only k at boundaries between distinct gain values are realisable by a cutoff
    boundary = np.concatenate(([True], g[1:] != g[:-1], [True]))  # k = 0..n
    ks = np.flatnonzero(boundary)
    acc = corr_k[ks] / n
    cost = cost_k[ks]
    strong_acc = sc.mean()
    feasible = acc >= strong_acc - allowance_pp / 100.0 - _EPS
    if not feasible.any():  # cannot happen: k=0 is always feasible
        raise RuntimeError("no feasible cutoff")
    best_cost = cost[feasible].min()
    cand = feasible & (cost <= best_cost + 1e-15 * max(1.0, abs(best_cost)))
    best_acc = acc[cand].max()
    cand &= acc >= best_acc - _EPS
    k = ks[np.flatnonzero(cand)[-1]]
    if k == 0:
        cutoff = -np.inf
    elif k == n:
        cutoff = np.inf
    else:
        cutoff = float((g[k - 1] + g[k]) / 2.0)
    return dict(allowance_pp=float(allowance_pp), cutoff=cutoff, n_calibration=int(n),
                calibration_accuracy=float(corr_k[k] / n), calibration_cost=float(cost_k[k]),
                calibration_cheap_fraction=float(k / n))


def evaluate(strong: np.ndarray, out: Outcomes) -> dict:
    """Metrics for a fixed set of routing decisions (True = GPT-5)."""
    n = len(strong)
    correct = np.where(strong, out.strong_correct, out.cheap_correct)
    cost = np.where(strong, out.strong_cost, out.cheap_cost)
    s_acc, c_acc = out.strong_correct.mean(), out.cheap_correct.mean()
    s_cost, c_cost = out.strong_cost.sum(), out.cheap_cost.sum()
    return dict(
        n=int(n),
        accuracy=float(correct.mean()),
        cost=float(cost.sum()),
        gross_savings_pct=float(100.0 * (1.0 - cost.sum() / s_cost)),
        accuracy_loss_pp=float(100.0 * (s_acc - correct.mean())),
        cheap_fraction=float(1.0 - strong.mean()),
        always_strong_accuracy=float(s_acc), always_strong_cost=float(s_cost),
        always_cheap_accuracy=float(c_acc), always_cheap_cost=float(c_cost),
        always_cheap_savings_pct=float(100.0 * (1.0 - c_cost / s_cost)),
        always_cheap_loss_pp=float(100.0 * (s_acc - c_acc)),
    )


def evaluate_by_benchmark(strong: np.ndarray, out: Outcomes, benchmark: np.ndarray) -> dict:
    per = {}
    for b in sorted(set(benchmark.tolist())):
        m = benchmark == b
        per[b] = evaluate(strong[m], out.take(m))
    macro = dict(
        n_benchmarks=len(per),
        gross_savings_pct=float(np.mean([v["gross_savings_pct"] for v in per.values()])),
        accuracy_loss_pp=float(np.mean([v["accuracy_loss_pp"] for v in per.values()])),
        accuracy=float(np.mean([v["accuracy"] for v in per.values()])),
    )
    return dict(per_benchmark=per, equal_benchmark_average=macro)


def bootstrap(strong: np.ndarray, out: Outcomes, groups: np.ndarray | None = None, n_boot: int = 2000,
              seed: int = 221, other: np.ndarray | None = None) -> dict:
    """Paired percentile bootstrap over questions (or duplicate groups) with decisions fixed.

    Captures sampling variability of the evaluation questions only: it does not include
    variability from refitting the predictor, re-choosing the cutoff, or model selection.
    If `other` (a second decision vector on the same questions) is given, also returns the
    interval for the difference in savings and loss (strong minus other)."""
    rng = np.random.default_rng(seed)
    n = len(strong)
    if groups is None:
        members = [np.array([i]) for i in range(n)]
    else:
        _, inv = np.unique(groups, return_inverse=True)
        members = [np.flatnonzero(inv == g) for g in range(inv.max() + 1)]
    # per-group sufficient statistics
    def stats(dec):
        corr = np.where(dec, out.strong_correct, out.cheap_correct).astype(float)
        cost = np.where(dec, out.strong_cost, out.cheap_cost)
        return np.array([[len(m), corr[m].sum(), cost[m].sum(), out.strong_correct[m].sum(), out.strong_cost[m].sum()]
                         for m in members])
    A = stats(strong)
    B = stats(other) if other is not None else None
    G = len(members)
    idx = rng.integers(0, G, size=(n_boot, G))

    def agg(M):
        T = M[idx].sum(axis=1)  # (n_boot, 5)
        sav = 100.0 * (1.0 - T[:, 2] / T[:, 4])
        loss = 100.0 * (T[:, 3] - T[:, 1]) / T[:, 0]
        return sav, loss
    sav, loss = agg(A)
    q = lambda x: [float(v) for v in np.quantile(x, [0.025, 0.975])]
    res = dict(n_boot=n_boot, seed=seed, n_groups=int(G), gross_savings_pct_ci95=q(sav), accuracy_loss_pp_ci95=q(loss),
               prob_loss_within={f"{a:g}pp": float(np.mean(loss <= a + 1e-9)) for a in ALLOWANCES_PP})
    if B is not None:
        sav_b, loss_b = agg(B)
        d = sav - sav_b
        res["diff_savings_pct_ci90"] = [float(v) for v in np.quantile(d, [0.05, 0.95])]
        res["diff_savings_pct_ci95"] = q(d)
        res["diff_loss_pp_ci95"] = q(loss - loss_b)
        res["prob_savings_higher"] = float(np.mean(d > 0))
    return res


def frontier(gain: np.ndarray, out: Outcomes, n_points: int = 201) -> list[dict]:
    """Cost/accuracy curve over cutoffs (for plots). Evaluated on whatever data is passed in."""
    cuts = candidate_cutoffs(gain)
    if len(cuts) > n_points:
        cuts = cuts[np.unique(np.linspace(0, len(cuts) - 1, n_points).round().astype(int))]
    pts = []
    for c in cuts:
        m = evaluate(route_strong(gain, c), out)
        pts.append(dict(cutoff=float(c), gross_savings_pct=m["gross_savings_pct"], accuracy_loss_pp=m["accuracy_loss_pp"],
                        cheap_fraction=m["cheap_fraction"]))
    return pts


def within_benchmark_shuffle(strong: np.ndarray, out: Outcomes, benchmark: np.ndarray, n_perm: int = 2000,
                             seed: int = 221) -> dict:
    """Diagnostic, not a baseline. Inside each benchmark, shuffle which questions go to Flash while
    keeping the number sent to Flash in that benchmark fixed. A router that only learned
    benchmark-level routing rates looks like its shuffles; one that picks better individual
    questions is more accurate than its shuffles at a similar cost."""
    rng = np.random.default_rng(seed)
    n = len(strong)
    s_corr, c_corr = out.strong_correct.astype(float), out.cheap_correct.astype(float)
    s_acc, s_cost = s_corr.mean(), out.strong_cost.sum()
    acc = np.zeros(n_perm)
    cost = np.zeros(n_perm)
    for b in np.unique(benchmark):
        idx = np.flatnonzero(benchmark == b)
        k = int((~strong[idx]).sum())  # questions of this benchmark sent to Flash
        if k == 0 or k == len(idx):   # nothing to shuffle
            acc += np.where(strong[idx], s_corr[idx], c_corr[idx]).sum()
            cost += np.where(strong[idx], out.strong_cost[idx], out.cheap_cost[idx]).sum()
            continue
        d_corr = c_corr[idx] - s_corr[idx]                     # change if a question goes to Flash
        d_cost = out.cheap_cost[idx] - out.strong_cost[idx]
        # random k-subsets: rank random keys, the k smallest go to Flash
        pick = np.argpartition(rng.random((n_perm, len(idx))), k - 1, axis=1)[:, :k]
        acc += s_corr[idx].sum() + d_corr[pick].sum(axis=1)
        cost += out.strong_cost[idx].sum() + d_cost[pick].sum(axis=1)
    acc /= n
    real = evaluate(strong, out)
    sav = 100.0 * (1.0 - cost / s_cost)
    loss = 100.0 * (s_acc - acc)
    q = lambda x: [float(v) for v in np.quantile(x, [0.025, 0.975])]
    return dict(
        n_perm=n_perm, seed=seed,
        router=dict(gross_savings_pct=real["gross_savings_pct"], accuracy_loss_pp=real["accuracy_loss_pp"]),
        shuffled_mean=dict(gross_savings_pct=float(sav.mean()), accuracy_loss_pp=float(loss.mean())),
        shuffled_range95=dict(gross_savings_pct=q(sav), accuracy_loss_pp=q(loss)),
        # positive = the router is more accurate / saves more than a policy with the same per-benchmark rates
        accuracy_advantage_pp=float(loss.mean() - real["accuracy_loss_pp"]),
        savings_advantage_pct=float(real["gross_savings_pct"] - sav.mean()),
        share_of_shuffles_at_least_as_accurate=float(np.mean(loss <= real["accuracy_loss_pp"] + 1e-12)),
        share_of_shuffles_saving_at_least_as_much=float(np.mean(sav >= real["gross_savings_pct"] - 1e-12)),
    )


def reweighted(strong: np.ndarray, out: Outcomes, benchmark: np.ndarray, focus: str, share: float) -> dict:
    """Workload-mix sensitivity: reweight questions so that `focus` makes up `share` of the
    questions, the other benchmarks keeping their observed proportions to each other.
    Decisions are fixed; nothing is refitted."""
    is_f = benchmark == focus
    obs = is_f.mean()
    if obs in (0.0, 1.0):
        raise ValueError(f"cannot reweight: observed share of {focus} is {obs}")
    w = np.where(is_f, share / obs, (1.0 - share) / (1.0 - obs))
    wm = lambda v: float((w * v).sum() / w.sum())
    corr = np.where(strong, out.strong_correct, out.cheap_correct)
    cost = np.where(strong, out.strong_cost, out.cheap_cost)
    s_acc, c_acc, s_cost, c_cost = wm(out.strong_correct), wm(out.cheap_correct), wm(out.strong_cost), wm(out.cheap_cost)
    return dict(focus=focus, share=share, observed_share=float(obs),
                gross_savings_pct=100.0 * (1.0 - wm(cost) / s_cost), accuracy_loss_pp=100.0 * (s_acc - wm(corr)),
                accuracy=wm(corr), cheap_fraction=wm(~strong), always_strong_accuracy=s_acc, always_cheap_accuracy=c_acc,
                always_cheap_savings_pct=100.0 * (1.0 - c_cost / s_cost), always_cheap_loss_pp=100.0 * (s_acc - c_acc),
                mean_cost_per_question_always_strong=s_cost)

