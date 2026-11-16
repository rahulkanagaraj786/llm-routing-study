"""Build data/processed/paired_v2.jsonl.gz from the raw LLMRouterBench files and
cross-check it against the paired file shipped in the handoff package."""
import gzip, json, sys, collections
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from router import data as D

rows, prov = D.build_paired()
sha = D.write_paired(rows)
C = collections.Counter
print("rows", len(rows), "| sha256", sha)

# --- cross-check against the handoff package (position-wise, per benchmark) ---
pkg_path = D.ROOT / "Handoff/Router_Development_Comparison/data/paired_full.jsonl.gz"
pkg = [json.loads(l) for l in gzip.open(pkg_path, "rt") if l.strip()]
mine = {(r["benchmark"], r["position"]): r for r in rows}
chk = C()
for p in pkg:
    m = mine[(p["benchmark"], p["strong_position"])]
    chk["rows"] += 1
    chk["prompt_equal"] += p["prompt"] == m["prompt"]
    chk["scores_equal"] += (p["strong_score"] == m["strong_score"] and p["cheap_score"] == m["cheap_score"])
    chk["costs_equal"] += (abs(p["strong_cost"] - m["strong_cost"]) < 1e-12 and abs(p["cheap_cost"] - m["cheap_cost"]) < 1e-12)
    pkg_elig = bool(p["measurement_valid"] and p["same_prompt"] and p["cheap_score"] in (0, 1) and p["strong_score"] in (0, 1) and p["cheap_cost"] > 0 and p["strong_cost"] > 0)
    chk["eligibility_equal"] += pkg_elig == m["eligible"]
    chk["pkg_eligible"] += pkg_elig
    if p["prompt"] != m["prompt"]:
        print("  prompt differs:", p["benchmark"], p["strong_position"], repr(p["prompt"][:80]), "|| raw:", repr(m["prompt"][:80]), "| pkg valid", p["measurement_valid"], "| mine eligible", m["eligible"])
print("package cross-check:", dict(chk))

by = collections.defaultdict(C)
for r in rows:
    b = by[r["benchmark"]]
    b["n"] += 1; b["eligible"] += r["eligible"]; b["eligible_recovered"] += r["eligible_recovered"]
    for reason in r["exclusion_reasons"]:
        b[reason] += 1
print(f"{'benchmark':14s} {'role':15s} {'n':>5s} {'elig':>5s} {'+recov':>6s}  exclusion reasons (a row can have several)")
for bench, b in by.items():
    extra = {k: v for k, v in b.items() if k not in ("n", "eligible", "eligible_recovered")}
    print(f"{bench:14s} {D.BENCHMARKS[bench]['role']:15s} {b['n']:5d} {b['eligible']:5d} {b['eligible_recovered']:6d}  {extra}")
corpus = [r for r in rows if r["role"] != "fresh"]
print("corpus rows", len(corpus), "eligible", sum(r["eligible"] for r in corpus), "eligible incl. recovered HLE costs", sum(r["eligible_recovered"] for r in corpus))
print("duplicate qids:", len(rows) - len({r["qid"] for r in rows}))
dq = [q for q, c in C(r["qid"] for r in rows).items() if c > 1]
for q in dq[:5]:
    print("  dup:", [(r["benchmark"], r["position"], r["eligible"], repr(r["prompt"][:60])) for r in rows if r["qid"] == q])

prov.update(paired_sha256=sha, paired_path=str(D.PAIRED_PATH.relative_to(D.ROOT)), n_rows=len(rows),
            archive=dict(hf_repo="NPULH/LLMRouterBench", hf_revision="0e5af1b84bf73437a01a1849c0f1d2468baa93fc",
                         file="bench-release.tar.gz", sha256=D.sha256_file(D.ROOT / "data/raw/bench-release.tar.gz"),
                         bytes=(D.ROOT / "data/raw/bench-release.tar.gz").stat().st_size),
            package_crosscheck=dict(chk), per_benchmark={k: dict(v) for k, v in by.items()})
out = D.ROOT / "data/processed/paired_v2_provenance.json"
out.write_text(json.dumps(prov, indent=2))
print("wrote", out.relative_to(D.ROOT))
