"""Exploratory data analysis: figures and numbers for the report's EDA section.

Figures 1-4 describe the recorded outcomes of the two LLMs on all eligible corpus questions
(baseline facts, the same ones the data audit reports; no router is involved).
Figures 5-6 relate prompt features to the outcome and use source-pool TRAINING questions only.

Writes results/eda/*.png and results/eda/eda_summary.json.
"""
import sys, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from router import data as D, splits as S

OUT = D.ROOT / "results" / "eda"
OUT.mkdir(parents=True, exist_ok=True)
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, DARK, LIGHT = "#2a78d6", "#eb6834", "#6f6e69", "#cfcec8"
plt.rcParams.update({"font.size": 11, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "text.color": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                     "savefig.facecolor": SURFACE})
NAMES = {"simpleqa": "SimpleQA", "mmlupro": "MMLU-Pro", "gpqa": "GPQA", "hle": "HLE", "livecodebench": "LiveCodeBench",
         "aime": "AIME", "livemathbench": "LiveMathBench"}
KIND = {"simpleqa": "short facts", "mmlupro": "multiple choice", "gpqa": "science, multiple choice", "hle": "expert questions",
        "livecodebench": "coding", "aime": "competition math", "livemathbench": "math"}


def clean(ax, grid_axis="x"):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, lw=1)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def footer(fig, text):
    import textwrap
    fig.text(0.01, 0.012, "\n".join(textwrap.wrap(text, int(fig.get_size_inches()[0] * 15.5))), fontsize=8.5, color=INK2,
             va="bottom", ha="left", linespacing=1.35)


rows = [r for r in D.read_paired() if r["eligible"] and r["role"] != "fresh"]
order = sorted(NAMES, key=lambda b: -sum(r["benchmark"] == b for r in rows))
summary = dict(n=len(rows), benchmarks={})


def stats(rs):
    g = np.array([r["strong_score"] for r in rs]); f = np.array([r["cheap_score"] for r in rs])
    gc = np.array([r["strong_cost"] for r in rs]); fc = np.array([r["cheap_cost"] for r in rs])
    words = np.array([len(r["prompt"].split()) for r in rs])
    return dict(n=len(rs), gpt5_acc=float(g.mean()), flash_acc=float(f.mean()),
                both_right=float(((g == 1) & (f == 1)).mean()), only_gpt5=float(((g == 1) & (f == 0)).mean()),
                only_flash=float(((g == 0) & (f == 1)).mean()), both_wrong=float(((g == 0) & (f == 0)).mean()),
                gpt5_cost_mean=float(gc.mean()), flash_cost_mean=float(fc.mean()), cost_ratio=float(gc.sum() / fc.sum()),
                flash_costlier_share=float((fc > gc).mean()),
                words_p10=float(np.quantile(words, .1)), words_median=float(np.median(words)), words_p90=float(np.quantile(words, .9)))


for b in order:
    summary["benchmarks"][b] = stats([r for r in rows if r["benchmark"] == b])
summary["all"] = stats(rows)
lab = lambda b: f"{NAMES[b]}  ({KIND[b]}, n = {summary['benchmarks'][b]['n']:,})"

# ---------------------------------------------------------------- Figure 1: outcome breakdown
keys = ["all"] + order
ylabels = [f"All questions  (n = {summary['all']['n']:,})"] + [lab(b) for b in order]
parts = [("both_right", "Both right", DARK, "white"), ("only_gpt5", "Only GPT-5 right", BLUE, "white"),
         ("only_flash", "Only Flash right", ORANGE, INK), ("both_wrong", "Both wrong", LIGHT, INK)]
fig, ax = plt.subplots(figsize=(11.5, 5.6), dpi=200)
y = np.arange(len(keys))[::-1]
left = np.zeros(len(keys))
for key, name, col, tcol in parts:
    vals = np.array([100 * (summary[k] if k == "all" else summary["benchmarks"][k])[key] for k in keys])
    ax.barh(y, vals, left=left, height=0.5, color=col, edgecolor=SURFACE, linewidth=2, label=name, zorder=3)
    for yy, l, v in zip(y, left, vals):
        if v >= 7:
            ax.text(l + v / 2, yy, f"{v:.0f}%", ha="center", va="center", fontsize=9.5, color=tcol, zorder=4)
    left += vals
ax.set_yticks(y); ax.set_yticklabels(ylabels, fontsize=10, color=INK)
ax.set_xlim(0, 100); ax.set_xlabel("Share of questions (%)")
ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=4, frameon=False, fontsize=9.5, labelcolor=INK, handlelength=1.2, columnspacing=1.6)
ax.set_title("How often do the two models agree?", loc="left", fontsize=13.5, color=INK, pad=30)
clean(ax); ax.grid(False)
footer(fig, "Recorded outcomes of GPT-5 and Gemini 2.5 Flash on the same questions (LLMRouterBench). Only the blue share needs GPT-5; "
            "everywhere else Flash gives the same result or a better one.")
fig.subplots_adjust(left=0.3, right=0.98, top=0.84, bottom=0.17)
fig.savefig(OUT / "eda1_outcome_breakdown.png"); plt.close(fig)

# ---------------------------------------------------------------- Figure 2: accuracy of each model
fig, ax = plt.subplots(figsize=(10.5, 5.0), dpi=200)
y = np.arange(len(order))[::-1]
for yy, b in zip(y, order):
    s = summary["benchmarks"][b]
    g, f = 100 * s["gpt5_acc"], 100 * s["flash_acc"]
    ax.plot([f, g], [yy, yy], color=LIGHT, lw=3, solid_capstyle="round", zorder=2)
    ax.plot(g, yy, "o", ms=10, color=BLUE, mec=SURFACE, mew=2, zorder=4, label="GPT-5" if yy == y[0] else None)
    ax.plot(f, yy, "o", ms=10, color=ORANGE, mec=SURFACE, mew=2, zorder=4, label="Gemini 2.5 Flash" if yy == y[0] else None)
    ax.text(max(g, f) + 2.2, yy, f"gap {g - f:+.0f}", va="center", fontsize=9.5, color=INK)
ax.set_yticks(y); ax.set_yticklabels([lab(b) for b in order], fontsize=10, color=INK)
ax.set_xlim(0, 104); ax.set_ylim(-0.6, len(order) - 0.4); ax.set_xlabel("Questions answered correctly (%)")
ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=9.5, labelcolor=INK)
ax.set_title("GPT-5 is stronger on every benchmark, by 6 to 26 points", loc="left", fontsize=12.5, color=INK, pad=30)
clean(ax)
footer(fig, "Share of questions each model answered correctly, by benchmark. The gap is in percentage points.")
fig.subplots_adjust(left=0.36, right=0.97, top=0.83, bottom=0.17)
fig.savefig(OUT / "eda2_accuracy_by_benchmark.png"); plt.close(fig)

# ---------------------------------------------------------------- Figure 3: cost per question (log scale)
fig, ax = plt.subplots(figsize=(10.5, 5.0), dpi=200)
for yy, b in zip(y, order):
    s = summary["benchmarks"][b]
    g, f = s["gpt5_cost_mean"], s["flash_cost_mean"]
    ax.plot([f, g], [yy, yy], color=LIGHT, lw=3, solid_capstyle="round", zorder=2)
    ax.plot(g, yy, "o", ms=10, color=BLUE, mec=SURFACE, mew=2, zorder=4, label="GPT-5" if yy == y[0] else None)
    ax.plot(f, yy, "o", ms=10, color=ORANGE, mec=SURFACE, mew=2, zorder=4, label="Gemini 2.5 Flash" if yy == y[0] else None)
    ax.text(g * 1.35, yy, f"{s['cost_ratio']:.0f}× the cost" if s["cost_ratio"] >= 10 else f"{s['cost_ratio']:.1f}× the cost", va="center", fontsize=9.5, color=INK)
ax.set_xscale("log"); ax.set_xlim(8e-5, 0.6)
ax.set_xticks([1e-4, 1e-3, 1e-2, 1e-1]); ax.set_xticklabels(["$0.0001", "$0.001", "$0.01", "$0.10"])
ax.set_yticks(y); ax.set_yticklabels([lab(b) for b in order], fontsize=10, color=INK)
ax.set_ylim(-0.6, len(order) - 0.4); ax.set_xlabel("Average recorded cost per question (log scale)")
ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=9.5, labelcolor=INK)
ax.set_title("GPT-5 costs more on every benchmark, by 2× to 91×", loc="left", fontsize=12.5, color=INK, pad=30)
clean(ax); ax.minorticks_off()
footer(fig, f"Average recorded cost per question. Each step on the axis is ten times the previous one. Flash cost more than GPT-5 on "
            f"{100 * summary['all']['flash_costlier_share']:.1f}% of questions, mostly in HLE.")
fig.subplots_adjust(left=0.36, right=0.97, top=0.83, bottom=0.19)
fig.savefig(OUT / "eda3_cost_by_benchmark.png"); plt.close(fig)

# ---------------------------------------------------------------- Figure 4: prompt length
fig, ax = plt.subplots(figsize=(10.5, 4.8), dpi=200)
for yy, b in zip(y, order):
    s = summary["benchmarks"][b]
    ax.plot([s["words_p10"], s["words_p90"]], [yy, yy], color=BLUE, lw=2, solid_capstyle="round", zorder=3)
    ax.plot(s["words_median"], yy, "o", ms=10, color=BLUE, mec=SURFACE, mew=2, zorder=4)
    ax.text(s["words_p90"] * 1.15, yy, f"median {s['words_median']:.0f} words", va="center", fontsize=9.5, color=INK)
ax.set_xscale("log"); ax.set_xlim(5, 3000)
ax.set_xticks([10, 30, 100, 300, 1000]); ax.set_xticklabels(["10", "30", "100", "300", "1,000"])
ax.set_yticks(y); ax.set_yticklabels([lab(b) for b in order], fontsize=10, color=INK)
ax.set_ylim(-0.6, len(order) - 0.4); ax.set_xlabel("Prompt length in words (log scale)")
ax.set_title("The benchmarks look very different before any model answers", loc="left", fontsize=12.5, color=INK, pad=14)
clean(ax); ax.minorticks_off()
footer(fig, "Dot: median prompt length. Line: the middle 80% of prompts (10th to 90th percentile). Prompts include each benchmark's instruction text.")
fig.subplots_adjust(left=0.36, right=0.97, top=0.88, bottom=0.2)
fig.savefig(OUT / "eda4_prompt_length.png"); plt.close(fig)

# ================= features against the outcome: source-pool TRAINING questions only =================
tab, trows = S.load_rows("eda:features_vs_outcome", splits=("train",), benchmarks=tuple(D.SELECTION_POOL),
                         exclude=lambda r: not S.in_source_pool(r))
yv = tab.outcomes.y.astype(float)
bench = tab.benchmark
summary["train_source_pool"] = dict(n=len(tab), mean_advantage=float(yv.mean()))


def mean_ci(v):
    m = float(v.mean()); se = float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0
    return m, m - 1.96 * se, m + 1.96 * se


# ---------------------------------------------------------------- Figure 5: MMLU-Pro subjects
subj = np.array([r["subject"] or "" for r in trows])
m = bench == "mmlupro"
S5 = sorted(((s, *mean_ci(yv[m & (subj == s)]), int((m & (subj == s)).sum())) for s in sorted(set(subj[m]))), key=lambda t: t[1])
summary["mmlupro_subjects_train"] = [dict(subject=s, advantage=a, lo=lo, hi=hi, n=n) for s, a, lo, hi, n in S5]
overall = float(yv[m].mean())
fig, ax = plt.subplots(figsize=(10.5, 5.6), dpi=200)
for i, (s, a, lo, hi, n) in enumerate(S5):
    ax.plot([100 * lo, 100 * hi], [i, i], color=BLUE, lw=2, solid_capstyle="round", zorder=3)
    ax.plot(100 * a, i, "o", ms=10, color=BLUE, mec=SURFACE, mew=2, zorder=4)
ax.axvline(100 * overall, color=INK2, lw=1, zorder=1)
ax.text(100 * overall, len(S5) - 0.35, f" MMLU-Pro average {100 * overall:.1f}", fontsize=9, color=INK2, va="center")
ax.axvline(0, color=GRID, lw=1.5, zorder=1)
ax.set_yticks(range(len(S5))); ax.set_yticklabels([f"{s.capitalize()}  (n = {n})" for s, a, lo, hi, n in S5], fontsize=10, color=INK)
ax.set_ylim(-0.6, len(S5) - 0.1)
ax.set_xlabel("GPT-5's advantage: extra questions it gets right per 100, compared with Flash")
ax.set_title("Within one benchmark, the subject says little about when GPT-5 helps", loc="left", fontsize=12.5, color=INK, pad=14)
clean(ax)
footer(fig, f"MMLU-Pro training questions in the source pool only (n = {int(m.sum()):,}; math and computer science are excluded by design). "
            "Dot: observed advantage. Line: 95% interval. The subject is named in the prompt, so a router can see it.")
fig.subplots_adjust(left=0.24, right=0.97, top=0.9, bottom=0.2)
fig.savefig(OUT / "eda5_advantage_by_subject.png"); plt.close(fig)

# ---------------------------------------------------------------- Figure 6: advantage by prompt length, within benchmark
words = np.array([len(p.split()) for p in tab.prompts])
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), dpi=200, sharey=True)
summary["advantage_by_length_train"] = {}
for ax, b in zip(axes, ("simpleqa", "mmlupro")):
    mb = bench == b
    qs = np.quantile(words[mb], [0.25, 0.5, 0.75])
    grp = np.digitize(words[mb], qs)
    pts = [mean_ci(yv[mb][grp == k]) for k in range(4)]
    summary["advantage_by_length_train"][b] = dict(quartile_cuts_words=[float(q) for q in qs],
                                                    advantage=[p[0] for p in pts], n=[int((grp == k).sum()) for k in range(4)])
    for k, (a, lo, hi) in enumerate(pts):
        ax.plot([k, k], [100 * lo, 100 * hi], color=BLUE, lw=2, solid_capstyle="round", zorder=3)
        ax.plot(k, 100 * a, "o", ms=10, color=BLUE, mec=SURFACE, mew=2, zorder=4)
    ax.axhline(100 * float(yv[mb].mean()), color=INK2, lw=1, zorder=1)
    ax.set_xticks(range(4)); ax.set_xticklabels(["shortest\nquarter", "2nd", "3rd", "longest\nquarter"], fontsize=9.5)
    ax.set_title(f"{NAMES[b]}  (n = {int(mb.sum()):,})", loc="left", fontsize=11, color=INK)
    ax.set_xlim(-0.5, 3.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=1); ax.set_axisbelow(True); ax.tick_params(length=0)
axes[0].set_ylabel("GPT-5's advantage per 100 questions")
fig.suptitle("Does prompt length tell us when GPT-5 helps?", x=0.01, ha="left", fontsize=12.5, color=INK)
footer(fig, "Source-pool training questions only, split into four equal groups by prompt length within each benchmark. "
            "Dot: observed advantage. Line: 95% interval. Horizontal line: that benchmark's average.")
fig.subplots_adjust(left=0.09, right=0.98, top=0.83, bottom=0.25, wspace=0.08)
fig.savefig(OUT / "eda6_advantage_by_length.png"); plt.close(fig)

(OUT / "eda_summary.json").write_text(json.dumps(summary, indent=2))
a = summary["all"]
print(f"all: n={a['n']} both right {100*a['both_right']:.1f}% only GPT-5 {100*a['only_gpt5']:.1f}% only Flash {100*a['only_flash']:.1f}% both wrong {100*a['both_wrong']:.1f}%")
print("subjects (advantage per 100):", [(s, round(100 * x, 1), n) for s, x, lo, hi, n in S5])
print("by length:", {b: [round(100 * x, 1) for x in v["advantage"]] for b, v in summary["advantage_by_length_train"].items()})
print("wrote", sorted(p.name for p in OUT.glob("*.png")))
