"""Development cost/accuracy figure from pooled out-of-fold predictions (selection pool only)."""
import sys, csv, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from router import splits as S, policy as P

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
from router import registry
_PALETTE = ["#2a78d6", "#eb6834", "#1baf7a"]  # validated together for up to three series
SERIES = {f: (_PALETTE[i], registry.label(f)) for i, f in enumerate(registry.names()[:3])}
BASE = S.D.ROOT / "results/dev_selection"
tab, _ = S.load_selection_dev("plot_dev")
pos = {q: i for i, q in enumerate(tab.qid)}

fig, ax = plt.subplots(figsize=(8.2, 5.2), dpi=160)
fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
XMAX = 3.5
for a in (1.0, 2.0):
    ax.axvline(a, color=GRID, lw=1, zorder=1)
    ax.text(a, 1.0, f"{a:g} pp allowance", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=8, color=INK2)
ends = []
for fam, (color, label) in SERIES.items():
    f = BASE / fam / "oof_predictions.csv"
    if not f.exists():
        continue
    rows = list(csv.DictReader(open(f)))
    g = np.zeros(len(tab)); d = {a: np.zeros(len(tab), bool) for a in ("0", "1", "2")}
    for r in rows:
        g[pos[r["qid"]]] = float(r["gain"])
        for a in d:
            d[a][pos[r["qid"]]] = r[f"route_gpt5_at_{a}pp"] == "1"
    fr = P.frontier(g, tab.outcomes, n_points=400)
    x = np.array([p["accuracy_loss_pp"] for p in fr]); y = np.array([p["gross_savings_pct"] for p in fr])
    keep = x <= XMAX
    ax.plot(x[keep], y[keep], color=color, lw=2, solid_capstyle="round", solid_joinstyle="round", label=label, zorder=3)
    for a in d:
        m = P.evaluate(d[a], tab.outcomes)
        ax.plot(m["accuracy_loss_pp"], m["gross_savings_pct"], "o", ms=8, color=color, mec=SURFACE, mew=2, zorder=4)
    ends.append((y[keep][np.argmax(x[keep])], label))
ends.sort()
for i, (yv, label) in enumerate(ends):  # direct labels at the line ends, nudged apart
    yv = max(yv, ends[i - 1][0] + 2.2) if i else yv
    ends[i] = (yv, label)
    ax.text(XMAX + 0.05, yv, label, fontsize=9, color=INK, va="center", ha="left")
ax.set_xlim(-0.3, XMAX); ax.set_ylim(0, None)
ax.set_xlabel("Accuracy loss vs always GPT-5 (percentage points)", color=INK2, fontsize=9)
ax.set_ylabel("Gross API cost saved vs always GPT-5 (%)", color=INK2, fontsize=9)
ax.set_title("Development cost–accuracy trade-off, source pool (out-of-fold, n = %d)" % len(tab), loc="left", fontsize=11, color=INK, pad=18)
ax.grid(axis="y", color=GRID, lw=1); ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(GRID)
ax.tick_params(colors=INK2, labelsize=8, length=0)
leg = ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK)
fig.text(0.01, 0.01, "Lines: every cutoff applied to pooled out-of-fold gains. Dots: cutoffs chosen on a calibration fold for 0, 1 and 2 pp, applied to unseen folds.\n"
                     "SimpleQA, GPQA and MMLU-Pro (without math and computer science) development rows only. Not a test result.", fontsize=7.5, color=INK2, va="bottom")
fig.subplots_adjust(left=0.08, right=0.83, top=0.88, bottom=0.2)
out = S.D.ROOT / "results/dev_selection/dev_tradeoff.png"
fig.savefig(out, facecolor=SURFACE)
print("wrote", out.relative_to(S.D.ROOT))
