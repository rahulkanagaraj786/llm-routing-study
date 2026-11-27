"""RQ2 workstream B, step 1: label-blind cross-dataset overlap audit.

Uses prompt/question text only (no outcomes, costs, or routing metrics).
Near-duplicates: same rule as src/router/splits.py — character 4-gram TF-IDF cosine >= 0.85.

  python rq2/audit_rq2_b_overlap.py

Writes under results/rq2/logistic/b_fewshot/:
  overlap_audit.json
  label_pool_qids.json   (cleaned LCB train qids allowed for k-sampling)
  eval_qids.json         (cleaned LCB test qids for B evaluation)
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from router import data as D, splits as S

OUT = D.ROOT / "results" / "rq2" / "logistic" / "b_fewshot"
A_PRED = D.ROOT / "results" / "rq2" / "logistic" / "predictions_rq2_transfer__code.csv"
THRESHOLD = S.NEAR_DUP_THRESHOLD
MIN_LABEL_POOL = 400


def main() -> None:
    man = S.read_manifest()
    # Text only: never touch outcomes / costs.
    rows = []
    for r in D.read_paired():
        if not r["eligible"] or r["qid"] not in man:
            continue
        m = man[r["qid"]]
        rows.append({
            "qid": r["qid"],
            "benchmark": r["benchmark"],
            "subject": r.get("subject"),
            "prompt": r["prompt"],
            "query": r["query"],
            "split": m["split"],
            "cluster_id_manifest": m["cluster_id"],
            "role": r["role"],
        })

    # Source-pool train/val for the audit: selection_pool without MMLU math/CS.
    # Do NOT apply boundary_duplicate_qids() here — those are exactly the source rows that
    # already share a near-dup cluster with a holdout (including LCB); excluding them would
    # hide the overlaps B needs to drop from the LCB label/eval sets.
    excluded_subj = S.SOURCE_POOL_EXCLUDED_MMLU_SUBJECTS
    source = [
        r for r in rows
        if r["role"] == "selection_pool"
        and r["split"] in ("train", "val")
        and not (r["benchmark"] == "mmlupro" and r["subject"] in excluded_subj)
    ]
    lcb = [r for r in rows if r["benchmark"] == "livecodebench"]
    lcb_train = [r for r in lcb if r["split"] == "train"]
    lcb_val = [r for r in lcb if r["split"] == "val"]
    lcb_test = [r for r in lcb if r["split"] == "test"]

    # Re-cluster source train/val + all LCB with the project near-dup rule (prompts only).
    combined = source + lcb
    cluster_of, pairs = S.near_duplicate_clusters(combined, threshold=THRESHOLD)

    source_qids = {r["qid"] for r in source}
    lcb_train_qids = {r["qid"] for r in lcb_train}
    lcb_test_qids = {r["qid"] for r in lcb_test}
    lcb_val_qids = {r["qid"] for r in lcb_val}

    members = defaultdict(set)
    for qid, cid in cluster_of.items():
        members[cid].add(qid)

    contaminated = []
    for cid, qs in members.items():
        has_source = bool(qs & source_qids)
        has_lcb = bool(qs & (lcb_train_qids | lcb_val_qids | lcb_test_qids))
        if has_source and has_lcb:
            contaminated.append({
                "cluster_id": cid,
                "n": len(qs),
                "source_qids": sorted(qs & source_qids),
                "lcb_train_qids": sorted(qs & lcb_train_qids),
                "lcb_val_qids": sorted(qs & lcb_val_qids),
                "lcb_test_qids": sorted(qs & lcb_test_qids),
            })

    drop_train_clusters = {
        cluster_of[q] for c in contaminated for q in c["lcb_train_qids"]
    }
    drop_test_clusters = {
        cluster_of[q] for c in contaminated for q in c["lcb_test_qids"]
    }

    excluded_train = sorted(q for q in lcb_train_qids if cluster_of[q] in drop_train_clusters)
    excluded_test = sorted(q for q in lcb_test_qids if cluster_of[q] in drop_test_clusters)
    label_pool = sorted(q for q in lcb_train_qids if cluster_of[q] not in drop_train_clusters)
    eval_qids = sorted(q for q in lcb_test_qids if cluster_of[q] not in drop_test_clusters)

    lcb_by_cluster = defaultdict(set)
    for r in lcb:
        lcb_by_cluster[r["cluster_id_manifest"]].add(r["split"])
    cross_split_clusters = {
        cid: sorted(splits) for cid, splits in lcb_by_cluster.items() if len(splits) > 1
    }

    if not A_PRED.exists():
        raise SystemExit(f"missing A predictions: {A_PRED.relative_to(D.ROOT)}")
    a_qids = set()
    with A_PRED.open() as f:
        header = f.readline()
        if not header.startswith("qid,"):
            raise SystemExit("unexpected A predictions header")
        for line in f:
            a_qids.add(line.split(",", 1)[0])
    missing_in_a = sorted(set(eval_qids) - a_qids)
    a_check = {
        "a_predictions_path": str(A_PRED.relative_to(D.ROOT)),
        "a_predictions_n": len(a_qids),
        "cleaned_eval_n": len(eval_qids),
        "cleaned_eval_missing_from_a_predictions": missing_in_a,
        "a_covers_cleaned_eval": len(missing_in_a) == 0,
    }

    ok_pool = len(label_pool) >= MIN_LABEL_POOL
    audit = {
        "purpose": "rq2_workstream_b_overlap_audit",
        "label_blind": True,
        "near_dup_rule": {
            "description": "character 4-gram TF-IDF cosine on normalized question text",
            "threshold": THRESHOLD,
            "function": "router.splits.near_duplicate_clusters",
            "dedup_text": "router.splits.dedup_text",
        },
        "inputs": {
            "source_pool_train_val_n": len(source),
            "lcb_train_n": len(lcb_train),
            "lcb_val_n": len(lcb_val),
            "lcb_test_n": len(lcb_test),
            "combined_n": len(combined),
            "n_pairs_at_threshold": len(pairs),
        },
        "contaminated_clusters_n": len(contaminated),
        "contaminated_clusters": contaminated,
        "exclusions": {
            "lcb_train_clusters_dropped": sorted(drop_train_clusters),
            "lcb_test_clusters_dropped": sorted(drop_test_clusters),
            "lcb_train_qids_excluded": excluded_train,
            "lcb_test_qids_excluded": excluded_test,
        },
        "frozen_sets": {
            "label_pool_n": len(label_pool),
            "eval_n": len(eval_qids),
            "lcb_val_unused_n": len(lcb_val_qids),
            "min_label_pool_required": MIN_LABEL_POOL,
            "label_pool_meets_min": ok_pool,
        },
        "lcb_manifest_cross_split_clusters": cross_split_clusters,
        "a_predictions_check": a_check,
        "status": "ok" if ok_pool and not missing_in_a and not cross_split_clusters else "needs_review",
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "overlap_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    (OUT / "label_pool_qids.json").write_text(json.dumps({
        "description": "LCB train qids allowed for B k-sampling after overlap exclusions",
        "n": len(label_pool),
        "qids": label_pool,
    }, indent=2) + "\n")
    (OUT / "eval_qids.json").write_text(json.dumps({
        "description": "LCB test qids for B evaluation after overlap exclusions; never labelled",
        "n": len(eval_qids),
        "qids": eval_qids,
    }, indent=2) + "\n")

    print(f"wrote {OUT.relative_to(D.ROOT)}/overlap_audit.json")
    print(f"  source train+val: {len(source)}")
    print(f"  LCB train/val/test: {len(lcb_train)}/{len(lcb_val)}/{len(lcb_test)}")
    print(f"  contaminated clusters: {len(contaminated)}")
    print(f"  label pool: {len(label_pool)} (need >= {MIN_LABEL_POOL}: {ok_pool})")
    print(f"  eval set: {len(eval_qids)}")
    print(f"  A predictions cover eval: {a_check['a_covers_cleaned_eval']}")
    print(f"  manifest cross-split LCB clusters: {len(cross_split_clusters)}")
    print(f"  status: {audit['status']}")
    if not ok_pool:
        sys.exit("label pool below minimum; revise k grid before unlock")
    if missing_in_a:
        sys.exit("cleaned eval qids missing from A's saved predictions")
    if cross_split_clusters:
        sys.exit("LCB manifest has clusters spanning splits; needs review")


if __name__ == "__main__":
    main()
