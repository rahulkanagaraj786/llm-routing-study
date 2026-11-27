"""RQ2 pipeline: transfer to unseen kinds of questions (docs/PROTOCOL.md section 8).

  python rq2/run_rq2_transfer.py <model> --plan           # question counts only; reads no outcomes
  python rq2/run_rq2_transfer.py <model> --dry-run        # rehearsal on source-pool development questions
  python rq2/run_rq2_transfer.py <model> --unlock-final   # the real run: ONCE per model

Workstreams A and A2, settled on 2026-11-26 (protocol log entry 23):

  A   One router trained on the source pool only (SimpleQA, GPQA, MMLU-Pro without math and
      computer science). Its cutoffs are tuned on the pool's validation split and frozen, then
      it is scored on the pool's locked test (reference), LiveCodeBench (primary), AIME +
      LiveMathBench, HLE, and HLE with repaired costs. It is also compared with the RQ1
      router's frozen decisions on the targets' test questions, and diagnosed: does a miss
      come from the cutoff or from the order of the predicted gains?
  A2  The same recipe trained without MMLU-Pro law, history and philosophy, scored on those
      subjects (new topics), on LiveCodeBench, math and HLE (new kinds of task), and on the
      rest of the pool's locked test (reference).

The settings come from the source-pool development run (results/rq2_dev_selection/<model>/).
Output: results/rq2/<model>/ (final_results_rq2.json, cutoffs, predictions).
Tables and figures: python rq2/report_rq2.py <model>
"""
import sys, csv, json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from router import data as D, splits as S, policy as P, registry, runlog, final

# Set to True on 2026-11-26, once workstreams A and A2 were written into docs/PROTOCOL.md
# section 8 (log entry 23).
DESIGN_SETTLED = True

args = sys.argv[1:]
pos_args = [a for a in args if not a.startswith("--")]
if len(pos_args) != 1:
    sys.exit(__doc__)
family = pos_args[0]
registry.check(family)
DRY, UNLOCK, PLAN = "--dry-run" in args, "--unlock-final" in args, "--plan" in args
if DRY + UNLOCK + PLAN != 1:
    sys.exit("choose exactly one of --plan, --dry-run or --unlock-final")

POOL = tuple(D.SELECTION_POOL)
TARGETS = {
    "code": dict(priority="primary transfer", benchmarks=("livecodebench",)),
    "math": dict(priority="secondary transfer", benchmarks=("aime", "livemathbench")),
    "hle": dict(priority="secondary transfer", benchmarks=("hle",)),
}
ALL_SPLITS = ("train", "val", "test")
HELD_OUT_SUBJECTS = ("law", "history", "philosophy")   # A2: MMLU-Pro's humanities subjects
REHEARSAL_SUBJECTS = ("biology", "health")             # their stand-ins in --dry-run
PRIMARY_ALLOWANCE = "3"                                # RQ2's primary target level
A, A2 = "rq2_transfer", "rq2_near_far"
N_SEALED_LOADS = 5                                     # unlocked loads made by one real run
man = S.read_manifest()


def subject_in(r, subjects):
    return r["benchmark"] == "mmlupro" and r["subject"] in subjects


def clusters_with(subjects):
    """Near-duplicate groups that contain a question of these subjects (any split, any eligibility)."""
    return {man[r["qid"]]["cluster_id"] for r in D.read_paired() if subject_in(r, subjects) and r["qid"] in man}


out_pool = lambda r: not S.in_source_pool(r)

if PLAN:
    c = S.plan_counts
    tot = lambda d: f"{sum(d.values()):5d}  {d}"
    cl = clusters_with(HELD_OUT_SUBJECTS)
    a2_out = lambda r: out_pool(r) or subject_in(r, HELD_OUT_SUBJECTS) or man[r["qid"]]["cluster_id"] in cl
    print("RQ2 workstream A: one router trained on the source pool only")
    print("  fit       ", tot(c(splits=("train",), benchmarks=POOL, exclude=out_pool)))
    print("  cutoffs   ", tot(c(splits=("val",), benchmarks=POOL, exclude=out_pool)))
    print("  reference: source-pool locked test", tot(c(splits=("test",), benchmarks=POOL, exclude=out_pool)))
    for name, t in TARGETS.items():
        print(f"  target {name} ({t['priority']})", tot(c(splits=ALL_SPLITS, benchmarks=t["benchmarks"])))
    print("  target hle, costs repaired", tot(c(splits=ALL_SPLITS, benchmarks=("hle",), use_recovered_cost=True)))
    print(f"RQ2 workstream A2: the same recipe without MMLU-Pro {', '.join(HELD_OUT_SUBJECTS)}")
    print("  fit       ", tot(c(splits=("train",), benchmarks=POOL, exclude=a2_out)))
    print("  cutoffs   ", tot(c(splits=("val",), benchmarks=POOL, exclude=a2_out)))
    print("  reference ", tot(c(splits=("test",), benchmarks=POOL, exclude=a2_out)))
    for s in HELD_OUT_SUBJECTS:
        n = c(splits=ALL_SPLITS, benchmarks=("mmlupro",), exclude=lambda r, s=s: out_pool(r) or not subject_in(r, (s,)))
        print(f"  held-out subject {s:10s}", tot(n))
    dup = c(splits=ALL_SPLITS, benchmarks=POOL,
            exclude=lambda r: out_pool(r) or subject_in(r, HELD_OUT_SUBJECTS) or man[r["qid"]]["cluster_id"] not in cl)
    print("  removed as near-duplicates of a held-out-subject question", tot(dup))
    print("  far targets: the same questions as in A")
    sys.exit(0)

if not DESIGN_SETTLED:
    sys.exit("RQ2 is on hold. Nothing was read or run. See rq2/README.md.")

OUT = D.ROOT / "results" / ("rq2_dryrun" if DRY else "rq2") / family
RESULT_FILE = OUT / "final_results_rq2.json"
if UNLOCK and RESULT_FILE.exists():
    sys.exit(f"{RESULT_FILE.relative_to(D.ROOT)} already exists: this runs once per model. "
             "Do not rerun it to compare alternatives.")
dev_path = D.ROOT / "results" / "rq2_dev_selection" / family / "summary.json"
if not dev_path.exists():
    sys.exit(f"{dev_path.relative_to(D.ROOT)} not found: run  python scripts/05_dev_compare.py {family} --source-pool  first.")
config = json.loads(dev_path.read_text())["final_config"]
# the RQ1 router's frozen decisions (a rehearsal file in --dry-run), for the "router that saw the benchmark" comparison
rq1_pred_path = (D.ROOT / "results" / ("final_dryrun" if DRY else "final") / family
                 / f"predictions_rq1__{'pseudo_test' if DRY else 'locked_test'}.csv")
print(f"model: {family}  study: RQ2 (A and A2)  settings: {config}  mode: {'DRY RUN' if DRY else 'FINAL'}", flush=True)
OUT.mkdir(parents=True, exist_ok=True)
results = dict(mode="dry_run" if DRY else "final", study="rq2", family=family, label=registry.label(family), config=config,
               dev_summary_sha256=D.sha256_file(dev_path), protocol_sha256=runlog.protocol_sha256(),
               env=runlog.env_info(), studies={}, analyses={})


def subset(pair, keep):
    tab, rows = pair
    idx = np.array([i for i, r in enumerate(rows) if keep(r)], dtype=int)
    return tab.take(idx), [rows[i] for i in idx]


def from_rows(rows):
    rows = sorted(rows, key=lambda r: r["qid"])
    return D.to_table(rows), rows


# ---------------------------------------------------------------------------------------------
# Questions
# ---------------------------------------------------------------------------------------------
if DRY:
    # Rehearsal on source-pool development questions only. Folds 0-2 train, fold 3 calibrates,
    # fold 4 stands in for the locked test. GPQA plays the far target, biology and health the
    # held-out subjects. Law, history and philosophy are left out entirely.
    tab, folds = S.load_selection_dev(f"rq2_dry_run:{family}")
    rows_by_qid = {r["qid"]: r for r in D.read_paired()}
    dev = subset((tab, [rows_by_qid[q] for q in tab.qid]), lambda r: not subject_in(r, HELD_OUT_SUBJECTS))
    fold = {q: int(f) for q, f in zip(tab.qid, folds)}
    near_subjects, far_bench, ref_name, far_primary = REHEARSAL_SUBJECTS, "gpqa", "pseudo_reference", "gpqa_as_holdout"
    tr = subset(dev, lambda r: fold[r["qid"]] <= 2 and r["benchmark"] != far_bench)
    va = subset(dev, lambda r: fold[r["qid"]] == 3 and r["benchmark"] != far_bench)
    ref = subset(dev, lambda r: fold[r["qid"]] == 4 and r["benchmark"] != far_bench)
    far = {far_primary: subset(dev, lambda r: r["benchmark"] == far_bench)}
    extra = {}
    in_test = lambda q: fold[q] == 4
else:
    kw = dict(unlock_final=True)
    tag = f"final:rq2:{family}"
    near_subjects, ref_name, far_primary = HELD_OUT_SUBJECTS, "source_pool_locked_test", "code"
    # The source pool's train and validation questions are development data: no unlock is needed.
    tr = S.load_rows(f"{tag}:train", splits=("train",), benchmarks=POOL, exclude=out_pool)
    va = S.load_rows(f"{tag}:val", splits=("val",), benchmarks=POOL, exclude=out_pool)
    ref = S.load_rows(f"{tag}:source_test", splits=("test",), benchmarks=POOL, exclude=out_pool, **kw)
    far = {name: S.load_rows(f"{tag}:target:{name}", splits=ALL_SPLITS, benchmarks=t["benchmarks"], **kw)
           for name, t in TARGETS.items()}
    extra = {"hle_costs_repaired": S.load_rows(f"{tag}:target:hle_repaired", splits=ALL_SPLITS, benchmarks=("hle",),
                                               unlock_final=True, use_recovered_cost=True)}
    in_test = lambda q: man[q]["split"] == "test"

held_out_benchmarks = {b for t in TARGETS.values() for b in t["benchmarks"]} | ({far_bench} if DRY else set())
assert not (set(tr[0].benchmark.tolist()) | set(va[0].benchmark.tolist())) & held_out_benchmarks

# A2: no question of the held-out subjects, nor any near-duplicate of one, in training, calibration or reference
cl = clusters_with(near_subjects)
a2_keep = lambda r: not subject_in(r, near_subjects) and man[r["qid"]]["cluster_id"] not in cl
tr2, va2, ref2 = subset(tr, a2_keep), subset(va, a2_keep), subset(ref, a2_keep)
near = from_rows([r for pair in (tr, va, ref) for r in pair[1] if subject_in(r, near_subjects)])
n_removed_dup = sum(1 for pair in (tr, va, ref) for r in pair[1] if not subject_in(r, near_subjects) and not a2_keep(r))
assert not any(subject_in(r, near_subjects) for pair in (tr2, va2, ref2) for r in pair[1])
assert len(near[0]) > 0 and not set(near[0].qid) & (set(tr2[0].qid) | set(va2[0].qid) | set(ref2[0].qid))
ref2_name = f"{ref_name}_without_held_out_subjects"
results["design"] = dict(protocol="docs/PROTOCOL.md section 8, log entry 23", rehearsal=DRY,
                         primary=dict(study=A, target=far_primary, allowance_pp=float(PRIMARY_ALLOWANCE)),
                         reference=ref_name, held_out_subjects=list(near_subjects), far_targets=list(far))

# ---------------------------------------------------------------------------------------------
# Fit, freeze the cutoffs, score (the shared step used by RQ1 too)
# ---------------------------------------------------------------------------------------------
A_targets = {ref_name: ref, **far, **extra}
results["studies"][A] = dict(final.run_one(family, config, A, tr[0], va[0], A_targets, OUT, man),
                             priority={ref_name: "reference", **{n: t["priority"] for n, t in TARGETS.items()},
                                       **{n: "sensitivity" for n in extra}})
A2_targets = {ref2_name: ref2, "held_out_subjects": near, **far}
results["studies"][A2] = dict(final.run_one(family, config, A2, tr2[0], va2[0], A2_targets, OUT, man),
                              held_out_subjects=list(near_subjects), removed_as_near_duplicates=n_removed_dup)


# ---------------------------------------------------------------------------------------------
# Analyses from the frozen decisions (section 8). Everything below reads the predictions just
# written, so it uses exactly the decisions on record.
# ---------------------------------------------------------------------------------------------
KEYS = P.KEYS
q95 = lambda x: [float(v) for v in np.quantile(x, [0.025, 0.975])]


def predictions(label, name, tab):
    with open(OUT / f"predictions_{label}__{name}.csv") as f:
        rows = list(csv.DictReader(f))
    assert [r["qid"] for r in rows] == list(tab.qid)
    g = np.array([float(r["gain"]) for r in rows])
    dec = {a: np.array([r[f"route_gpt5_at_{a}pp"] == "1" for r in rows]) for a in KEYS}
    cut = results["studies"][label]["cutoffs"]
    assert all((P.route_strong(g, cut[a]["cutoff"]) == dec[a]).all() for a in KEYS)   # gains read back exactly
    return g, dec


def groups_of(tab):
    return np.array([man[q]["cluster_id"] for q in tab.qid])


def brief(m, b):
    return dict(gross_savings_pct=m["gross_savings_pct"], accuracy_loss_pp=m["accuracy_loss_pp"], sent_to_flash=m["cheap_fraction"],
                gross_savings_pct_ci95=b["gross_savings_pct_ci95"], accuracy_loss_pp_ci95=b["accuracy_loss_pp_ci95"])


def paired(dec_1, dec_2, out, groups, keys):
    """The same questions under two sets of frozen decisions: first minus second."""
    res = {}
    for a in keys:
        m1, m2 = P.evaluate(dec_1[a], out), P.evaluate(dec_2[a], out)
        b = P.bootstrap(dec_1[a], out, groups=groups, other=dec_2[a])
        res[a] = dict(first=brief(m1, b), second=brief(m2, P.bootstrap(dec_2[a], out, groups=groups)),
                      diff_savings_pct=m1["gross_savings_pct"] - m2["gross_savings_pct"],
                      diff_loss_pp=m1["accuracy_loss_pp"] - m2["accuracy_loss_pp"],
                      diff_savings_pct_ci95=b["diff_savings_pct_ci95"], diff_loss_pp_ci95=b["diff_loss_pp_ci95"],
                      always_strong_accuracy=m1["always_strong_accuracy"], always_cheap_accuracy=m1["always_cheap_accuracy"])
    return dict(n=len(out), n_groups=len(set(groups.tolist())), by_allowance=res)


def diagnose(g, tab, cutoffs):
    """Ranking versus cutoff. The hindsight cutoff is chosen on this set's own outcomes: diagnostic only."""
    out = tab.outcomes
    y = out.y
    only = y == 1
    ceil = P.evaluate(y > 0, out)
    per = {}
    for a, c in cutoffs.items():
        fz = P.evaluate(P.route_strong(g, c["cutoff"]), out)
        h = P.select_cutoff(g, out, float(a))
        hm = P.evaluate(P.route_strong(g, h["cutoff"]), out)
        per[a] = dict(sent_to_flash_on_source_validation=c["calibration_cheap_fraction"], frozen_cutoff=c["cutoff"],
                      frozen=dict(sent_to_flash=fz["cheap_fraction"], gross_savings_pct=fz["gross_savings_pct"], accuracy_loss_pp=fz["accuracy_loss_pp"]),
                      hindsight=dict(cutoff=h["cutoff"], sent_to_flash=hm["cheap_fraction"], gross_savings_pct=hm["gross_savings_pct"],
                                     accuracy_loss_pp=hm["accuracy_loss_pp"]))
    return dict(n=len(tab), mean_gain=float(g.mean()), gain_quantiles=[float(v) for v in np.quantile(g, [0.05, 0.25, 0.5, 0.75, 0.95])],
                mean_y=float(y.mean()), share_only_gpt5_right=float(only.mean()),
                corr_gain_y=float(np.corrcoef(g, y)[0, 1]) if g.std() > 0 and y.std() > 0 else None,
                auc_only_gpt5_right=float(roc_auc_score(only, g)) if 0 < only.sum() < len(y) else None,
                ceiling=dict(gross_savings_pct=ceil["gross_savings_pct"], accuracy_loss_pp=ceil["accuracy_loss_pp"], sent_to_flash=ceil["cheap_fraction"]),
                by_allowance=per)


def draws(dec, out, groups, rng, n_boot=2000):
    """Bootstrap draws of savings and loss, resampling duplicate groups (as policy.bootstrap does)."""
    _, inv = np.unique(groups, return_inverse=True)
    G = int(inv.max()) + 1
    corr = np.where(dec, out.strong_correct, out.cheap_correct).astype(float)
    cost = np.where(dec, out.strong_cost, out.cheap_cost)
    M = np.column_stack([np.bincount(inv, minlength=G).astype(float), np.bincount(inv, corr, G), np.bincount(inv, cost, G),
                         np.bincount(inv, out.strong_correct.astype(float), G), np.bincount(inv, out.strong_cost, G)])
    T = M[rng.integers(0, G, size=(n_boot, G))].sum(axis=1)
    return 100.0 * (1.0 - T[:, 2] / T[:, 4]), 100.0 * (T[:, 3] - T[:, 1]) / T[:, 0]


def independent_differences(sets, pairs, keys, seed=221):
    """Differences between disjoint sets of questions, each resampled on its own."""
    rng = np.random.default_rng(seed)
    d = {name: {a: draws(dec[a], out, gr, rng) for a in keys} for name, (dec, out, gr) in sets.items()}
    pt = {name: {a: P.evaluate(dec[a], out) for a in keys} for name, (dec, out, gr) in sets.items()}
    return {f"{x}_minus_{y}": {a: dict(diff_savings_pct=pt[x][a]["gross_savings_pct"] - pt[y][a]["gross_savings_pct"],
                                       diff_loss_pp=pt[x][a]["accuracy_loss_pp"] - pt[y][a]["accuracy_loss_pp"],
                                       diff_savings_pct_ci95=q95(d[x][a][0] - d[y][a][0]),
                                       diff_loss_pp_ci95=q95(d[x][a][1] - d[y][a][1]))
                                for a in keys}
            for x, y in pairs}


def reading(m, a):
    """The reading rule of section 8, fixed before the run."""
    t, loss = float(a), m["accuracy_loss_pp"]
    lo, slo = m["bootstrap"]["accuracy_loss_pp_ci95"][0], m["bootstrap"]["gross_savings_pct_ci95"][0]
    within = loss <= t + 1e-9
    acc = "within the target" if within else ("clearly exceeded" if lo > t + 1e-9 else "exceeded, not clearly")
    return dict(accuracy=acc, savings="clear savings" if slo > 0 else "no clear savings", holds=bool(within and slo > 0))


tabs = {A: {n: p[0] for n, p in A_targets.items()}, A2: {n: p[0] for n, p in A2_targets.items()}}
pred = {s: {n: predictions(s, n, t) for n, t in tabs[s].items()} for s in (A, A2)}
for s in (A, A2):
    for n, t in results["studies"][s]["targets"].items():
        for a, m in t["allowances"].items():
            m["reading"] = reading(m, a)

# 1. The router that saw the benchmark: RQ1's frozen decisions on the targets' test questions
if rq1_pred_path.exists():
    with open(rq1_pred_path) as f:
        rq1_rows = {r["qid"]: r for r in csv.DictReader(f)}
    rq1_keys = [a for a in KEYS if f"route_gpt5_at_{a}pp" in next(iter(rq1_rows.values()))]
    cmp = {}
    for name in far:
        tab = tabs[A][name]
        m = np.array([in_test(q) for q in tab.qid])
        sub = tab.take(np.flatnonzero(m))
        missing = [q for q in sub.qid if q not in rq1_rows]
        assert not missing, f"{len(missing)} {name} test questions have no RQ1 decision"
        d_rq1 = {a: np.array([rq1_rows[q][f"route_gpt5_at_{a}pp"] == "1" for q in sub.qid]) for a in rq1_keys}
        d_rq2 = {a: pred[A][name][1][a][m] for a in rq1_keys}
        cmp[name] = paired(d_rq2, d_rq1, sub.outcomes, groups_of(sub), rq1_keys)
    results["analyses"]["transfer_router_vs_rq1_router"] = dict(
        note="first = RQ2 transfer router (never saw the benchmark), second = RQ1 router (trained on its train questions); "
             "the targets' test-split questions only; paired bootstrap over duplicate groups, decisions fixed",
        rq1_predictions=str(rq1_pred_path.relative_to(D.ROOT)), rq1_predictions_sha256=D.sha256_file(rq1_pred_path),
        by_target=cmp)
else:
    results["analyses"]["transfer_router_vs_rq1_router"] = dict(note=f"{rq1_pred_path.relative_to(D.ROOT)} not found; not computed")

# 2. Ranking versus cutoff
results["analyses"]["ranking_vs_cutoff"] = {
    s: {n: diagnose(pred[s][n][0], tabs[s][n], results["studies"][s]["cutoffs"]) for n in tabs[s] if n not in extra}
    for s in (A, A2)}
results["analyses"]["ranking_vs_cutoff"]["note"] = (
    "hindsight = the cutoff the section 6 rule would choose on the set's own outcomes; it uses the outcomes being scored, "
    "so it is a diagnostic and never the router's result")

# 3. A2: new topics versus new kinds of task (independent bootstrap for differences between disjoint sets)
sets = {k: (pred[A2][k][1], tabs[A2][k].outcomes, groups_of(tabs[A2][k])) for k in (ref2_name, "held_out_subjects", far_primary)}
results["analyses"]["near_vs_far"] = dict(
    router=A2, far_target=far_primary,
    differences=independent_differences(sets, [("held_out_subjects", ref2_name), (far_primary, ref2_name),
                                               ("held_out_subjects", far_primary)], KEYS))

# 4. The held-out subjects one by one (descriptive)
nt, nrows = near
g_near, d_near = pred[A2]["held_out_subjects"]
by_sub = {}
for s in near_subjects:
    m = np.array([r["subject"] == s for r in nrows])
    sub = nt.take(np.flatnonzero(m))
    by_sub[s] = dict(n=int(m.sum()), allowances={
        a: brief(P.evaluate(d_near[a][m], sub.outcomes), P.bootstrap(d_near[a][m], sub.outcomes, groups=groups_of(sub)))
        for a in KEYS})
    by_sub[s]["always_strong_accuracy"] = float(sub.outcomes.strong_correct.mean())
    by_sub[s]["always_cheap_accuracy"] = float(sub.outcomes.cheap_correct.mean())
results["analyses"]["held_out_subjects_by_subject"] = by_sub

# 5. Seen versus unseen subjects: the held-out subjects' test questions under A's router and A2's router
m = np.array([in_test(q) for q in nt.qid])
sub = nt.take(np.flatnonzero(m))
pos_ref = {q: i for i, q in enumerate(tabs[A][ref_name].qid)}
d_seen = {a: pred[A][ref_name][1][a][[pos_ref[q] for q in sub.qid]] for a in KEYS}
d_unseen = {a: d_near[a][m] for a in KEYS}
results["analyses"]["seen_vs_unseen_subjects"] = dict(
    note="first = A2 router (never saw these subjects), second = A router (trained with them); test-split questions of the "
         "held-out subjects; paired bootstrap over duplicate groups", **paired(d_unseen, d_seen, sub.outcomes, groups_of(sub), KEYS))

# The primary result (section 8), and the record of sealed reads
pa = PRIMARY_ALLOWANCE
pick = lambda t: dict(n=t["n"], **brief(t["allowances"][pa], t["allowances"][pa]["bootstrap"]),
                      prob_loss_within=t["allowances"][pa]["bootstrap"]["prob_loss_within"][f"{pa}pp"],
                      within_allowance=t["allowances"][pa]["within_allowance"], reading=t["allowances"][pa]["reading"])
results["primary"] = dict(description=f"RQ2 primary: the frozen transfer router on {far_primary} at the {pa}-point target level, "
                                      "next to the same router on its source-pool reference",
                          study=A, target=far_primary, allowance_pp=float(pa),
                          result=pick(results["studies"][A]["targets"][far_primary]),
                          reference=pick(results["studies"][A]["targets"][ref_name]))
if not DRY:
    lines = S.ACCESS_LOG.read_text().splitlines() if S.ACCESS_LOG.exists() else []
    sealed = [ln for ln in lines if "unlock_final=True" in ln]
    mine = [ln for ln in sealed if ln.split("\t")[1].startswith(f"final:rq2:{family}:")]
    results["sealed_reads"] = dict(this_run=mine[-N_SEALED_LOADS:], earlier_rq2_reads_on_this_machine=mine[:-N_SEALED_LOADS],
                                   other_sealed_reads_on_this_machine=len(sealed) - len(mine))
RESULT_FILE.write_text(json.dumps(results, indent=2))

p = results["primary"]
for k in ("result", "reference"):
    v = p[k]
    print(f"PRIMARY {k:9s} n={v['n']:5d}  {pa} pp: savings {v['gross_savings_pct']:.1f}% "
          f"({v['gross_savings_pct_ci95'][0]:.1f} to {v['gross_savings_pct_ci95'][1]:.1f}), loss {v['accuracy_loss_pp']:.2f} pp "
          f"({v['accuracy_loss_pp_ci95'][0]:.2f} to {v['accuracy_loss_pp_ci95'][1]:.2f}), P(within) {v['prob_loss_within']:.2f}, "
          f"to Flash {100 * v['sent_to_flash']:.1f}% | {v['reading']['accuracy']}, {v['reading']['savings']}")
print("wrote", RESULT_FILE.relative_to(D.ROOT))
