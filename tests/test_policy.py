"""Evaluator checks on synthetic data only (no study outcomes are read)."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from router.data import Outcomes
from router import policy as P


def synth(n, rng, ties=False):
    gain = rng.normal(size=n)
    if ties:
        gain = np.round(gain, 1)
    sc = (rng.random(n) < 0.6).astype(int)
    cc = (rng.random(n) < 0.45).astype(int)
    s_cost = rng.gamma(2.0, 0.01, n)
    c_cost = rng.gamma(2.0, 0.004, n)  # sometimes larger than the strong cost, as in the real data
    return gain, Outcomes(sc, cc, s_cost, c_cost)


def brute(gain, out, allowance_pp):
    """The pilot's loop: cut at -inf and at each observed value, strict cost improvement."""
    best = None
    for tau in np.r_[-np.inf, np.unique(gain)]:
        strong = gain > tau
        acc = np.where(strong, out.strong_correct, out.cheap_correct).mean()
        cost = np.where(strong, out.strong_cost, out.cheap_cost).sum()
        if acc >= out.strong_correct.mean() - allowance_pp / 100 - 1e-12 and (best is None or cost < best[0] - 1e-15):
            best = (cost, acc)
    return best


def test_select_matches_bruteforce():
    rng = np.random.default_rng(0)
    for trial in range(60):
        gain, out = synth(int(rng.integers(5, 400)), rng, ties=trial % 2 == 0)
        for a in (0.0, 1.0, 2.0, 5.0):
            sel = P.select_cutoff(gain, out, a)
            m = P.evaluate(P.route_strong(gain, sel["cutoff"]), out)
            cost, acc = brute(gain, out, a)
            assert abs(m["cost"] - cost) < 1e-12, (trial, a, m["cost"], cost)
            assert m["accuracy"] >= acc - 1e-12
            assert abs(m["cost"] - sel["calibration_cost"]) < 1e-12 and abs(m["accuracy"] - sel["calibration_accuracy"]) < 1e-12
            assert m["accuracy_loss_pp"] <= a + 1e-9


def test_endpoints_and_definitions():
    rng = np.random.default_rng(1)
    gain, out = synth(300, rng)
    all_strong = P.evaluate(P.route_strong(gain, -np.inf), out)
    all_cheap = P.evaluate(P.route_strong(gain, np.inf), out)
    assert all_strong["cheap_fraction"] == 0 and abs(all_strong["gross_savings_pct"]) < 1e-12 and abs(all_strong["accuracy_loss_pp"]) < 1e-12
    assert all_cheap["cheap_fraction"] == 1
    assert abs(all_cheap["gross_savings_pct"] - all_cheap["always_cheap_savings_pct"]) < 1e-9
    assert abs(all_cheap["accuracy_loss_pp"] - 100 * (out.strong_correct.mean() - out.cheap_correct.mean())) < 1e-9
    # strict inequality: gain exactly equal to the cutoff goes to Flash
    assert not P.route_strong(np.array([0.5]), 0.5)[0] and P.route_strong(np.array([0.5000001]), 0.5)[0]
    # an oracle gain (the true Y) with zero allowance never loses accuracy
    y = out.y.astype(float)
    sel = P.select_cutoff(y, out, 0.0)
    assert P.evaluate(P.route_strong(y, sel["cutoff"]), out)["accuracy_loss_pp"] <= 1e-9


def test_per_benchmark_and_bootstrap():
    rng = np.random.default_rng(2)
    gain, out = synth(500, rng)
    bench = np.array(["a"] * 200 + ["b"] * 300)
    strong = P.route_strong(gain, 0.0)
    pooled = P.evaluate(strong, out)
    per = P.evaluate_by_benchmark(strong, out, bench)["per_benchmark"]
    assert abs(per["a"]["cost"] + per["b"]["cost"] - pooled["cost"]) < 1e-9
    assert abs((per["a"]["accuracy"] * 200 + per["b"]["accuracy"] * 300) / 500 - pooled["accuracy"]) < 1e-12
    b = P.bootstrap(strong, out, n_boot=500, other=P.route_strong(gain, -np.inf))
    lo, hi = b["gross_savings_pct_ci95"]
    assert lo <= pooled["gross_savings_pct"] <= hi
    lo, hi = b["accuracy_loss_pp_ci95"]
    assert lo <= pooled["accuracy_loss_pp"] <= hi
    assert b["diff_savings_pct_ci95"][0] > 0  # routing some questions to Flash saves vs always-strong here
    g = P.bootstrap(strong, out, groups=np.arange(500) // 2, n_boot=200)
    assert g["n_groups"] == 250


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
