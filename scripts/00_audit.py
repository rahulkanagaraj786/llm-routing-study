"""Data audit: regenerates the numbers quoted in docs/DATA_AUDIT.md into results/audit/data_audit.json.

Reports baseline facts about the two LLMs (accuracy, cost, missingness) per benchmark. These are
properties of the recorded data, not router results. Run after scripts/01_build_paired.py."""
import sys, json, glob, collections
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from router import data as D

rows = D.read_paired()
C = collections.Counter
A = dict(paired_sha256=D.sha256_file(D.PAIRED_PATH), n_rows=len(rows))

# 1. model identifiers and file-level metadata
prov = json.loads((D.ROOT / "data/processed/paired_v2_provenance.json").read_text())
A["model_names_in_files"] = sorted({f.get("model_name") for f in prov["files"] if f.get("model_name")})
A["fingerprints_match"] = {b: v["fingerprint_match"] for b, v in prov["benchmarks"].items()}
A["package_crosscheck"] = prov["package_crosscheck"]

# 2. per-benchmark baseline table (eligible rows)
tab = {}
for b in D.BENCHMARKS:
    R = [r for r in rows if r["benchmark"] == b]; E = [r for r in R if r["eligible"]]
    y = np.array([r["strong_score"] - r["cheap_score"] for r in E])
    sc, cc = np.array([r["strong_cost"] for r in E]), np.array([r["cheap_cost"] for r in E])
    tab[b] = dict(role=D.BENCHMARKS[b]["role"], n_records=len(R), n_eligible=len(E),
                  exclusions=dict(C(x for r in R for x in r["exclusion_reasons"])),
                  gpt5_accuracy=float(np.mean([r["strong_score"] for r in E])), flash_accuracy=float(np.mean([r["cheap_score"] for r in E])),
                  y_plus=int((y == 1).sum()), y_zero=int((y == 0).sum()), y_minus=int((y == -1).sum()),
                  both_correct=int(sum(r["strong_score"] == 1 and r["cheap_score"] == 1 for r in E)),
                  both_wrong=int(sum(r["strong_score"] == 0 and r["cheap_score"] == 0 for r in E)),
                  gpt5_cost=float(sc.sum()), flash_cost=float(cc.sum()), cost_ratio=float(sc.sum() / cc.sum()),
                  flash_costlier_rows=int((cc > sc).sum()), median_prompt_chars=float(np.median([len(r["prompt"]) for r in E])))
A["per_benchmark"] = tab
E = [r for r in rows if r["eligible"] and r["role"] != "fresh"]
A["corpus"] = dict(n_eligible=len(E), gpt5_accuracy=float(np.mean([r["strong_score"] for r in E])), flash_accuracy=float(np.mean([r["cheap_score"] for r in E])),
                   gpt5_cost=float(sum(r["strong_cost"] for r in E)), flash_cost=float(sum(r["cheap_cost"] for r in E)),
                   y_counts=dict(C(str(int(r["strong_score"] - r["cheap_score"])) for r in E)),
                   flash_costlier_rows=int(sum(r["cheap_cost"] > r["strong_cost"] for r in E)),
                   n_eligible_with_recovered_hle_costs=sum(r["eligible_recovered"] for r in rows if r["role"] != "fresh"))

# 3. recorded cost vs the price schedule implied by the release
price = {}
for b in D.BENCHMARKS:
    for who, model in (("strong", D.STRONG), ("cheap", D.CHEAP)):
        R = [r for r in rows if r["benchmark"] == b and r[f"{who}_cost"] > 0 and r[f"{who}_cost_list"]]
        ratio = np.array([r[f"{who}_cost"] / r[f"{who}_cost_list"] for r in R])
        price[f"{b}:{model}"] = dict(n=len(R), recorded_total=float(sum(r[f"{who}_cost"] for r in R)), list_price_total=float(sum(r[f"{who}_cost_list"] for r in R)),
                                     share_off_by_more_than_1pct=float(np.mean(np.abs(ratio - 1) > 0.01)), min_ratio=float(ratio.min()), median_ratio=float(np.median(ratio)))
A["recorded_vs_list_price"] = price
A["implied_price_per_million_tokens"] = {m: dict(input=p[0] * 1e6, output=p[1] * 1e6) for m, p in D.IMPLIED_PRICE_PER_TOKEN.items()}

# 4. HLE missing GPT-5 costs
H = [r for r in rows if r["benchmark"] == "hle"]
miss = [r for r in H if r["strong_cost"] <= 0]
A["hle_missing_gpt5_cost"] = dict(n=len(miss), recoverable_from_subset_500=sum(r["strong_cost_sister"] is not None for r in miss),
                                  gpt5_accuracy_missing=float(np.mean([r["strong_score"] for r in miss])),
                                  gpt5_accuracy_present=float(np.mean([r["strong_score"] for r in H if r["strong_cost"] > 0])),
                                  first_raw_index=min(r["raw_index"] for r in miss), last_raw_index=max(r["raw_index"] for r in miss))

# 5. score semantics
sem = {}
for b, meta in D.BENCHMARKS.items():
    f = glob.glob(str(D.RAW_DIR / meta["dataset"] / meta["split"] / D.STRONG / "*.json"))[0]
    R = json.load(open(f))["records"]
    sem[b] = dict(score_values=sorted({r["score"] for r in R}), distinct_predictions=len({str(r["prediction"]) for r in R}),
                  top_prediction_score_pairs=[[str(p)[:12], s, n] for (p, s), n in C((str(r["prediction"])[:12], r["score"]) for r in R).most_common(3)])
A["score_semantics"] = sem

# 6. other GPT-5 / Flash pairs in the release that are not used, and why
A["unused_pairs"] = {"arenahard (+3 subsets)": "scores take values 0, 0.5, 1: not binary correctness",
                     "tau2": "the 'prompt' field is a test description, not the model input; multi-turn agent task",
                     "hle/subset_500, simpleqa/subset_500, mmlupro/test_1000": "subsets of the larger splits already used (same questions)",
                     "humaneval": "no gpt-5 or gemini-2.5-flash run (gpt-5-chat only)"}
A["mmlupro_subjects"] = dict(C(r["subject"] for r in rows if r["benchmark"] == "mmlupro").most_common())
man_meta = json.loads((D.ROOT / "manifests/split_manifest_v1.meta.json").read_text())
A["near_duplicates"] = man_meta["near_duplicates"]
(D.ROOT / "results/audit").mkdir(parents=True, exist_ok=True)
(D.ROOT / "results/audit/data_audit.json").write_text(json.dumps(A, indent=2))

print(f"{'benchmark':14s} {'elig':>5s} {'GPT-5':>6s} {'Flash':>6s} {'Y=+1':>5s} {'Y=0':>5s} {'Y=-1':>5s} {'GPT-5 $':>8s} {'Flash $':>8s} {'ratio':>6s} {'Flash>GPT5':>10s}")
for b, t in tab.items():
    print(f"{b:14s} {t['n_eligible']:5d} {100*t['gpt5_accuracy']:6.1f} {100*t['flash_accuracy']:6.1f} {t['y_plus']:5d} {t['y_zero']:5d} {t['y_minus']:5d} {t['gpt5_cost']:8.2f} {t['flash_cost']:8.2f} {t['cost_ratio']:6.1f} {t['flash_costlier_rows']:10d}")
print("corpus:", json.dumps(A["corpus"]))
print("cost files off the implied price schedule:", {k: (round(v["recorded_total"], 2), round(v["list_price_total"], 2), round(v["share_off_by_more_than_1pct"], 3)) for k, v in price.items() if v["share_off_by_more_than_1pct"] > 0.005})
print("HLE missing GPT-5 cost:", A["hle_missing_gpt5_cost"])
