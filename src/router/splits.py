"""Near-duplicate grouping, deterministic split manifests, and guarded loaders.

Splits are derived from SHA-256 hashes of a fixed salt and the cluster id, so they
do not depend on a random-number generator, library version, or platform, and a
question's assignment does not change if eligibility rules change later.
"""
from __future__ import annotations

import csv
import functools
import datetime as _dt
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import data as D

SALT = "STA221-llm-router-split-v1"
TEST_FRACTION = 0.20
VAL_FRACTION = 0.20  # of all rows; remaining 0.60 is train
N_FOLDS = 5
NEAR_DUP_THRESHOLD = 0.85  # cosine on character 4-gram TF-IDF of the normalized question text
MANIFEST_PATH = D.ROOT / "manifests" / "split_manifest_v1.csv"
META_PATH = D.ROOT / "manifests" / "split_manifest_v1.meta.json"
ACCESS_LOG = D.ROOT / "logs" / "data_access.log"

_WS = re.compile(r"\s+")


def _h(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def dedup_text(r: dict) -> str:
    """Question text without the benchmark's instruction template where available."""
    q = r["query"] if len(r["query"]) >= 20 else r["prompt"]
    return _WS.sub(" ", q.lower()).strip()


def near_duplicate_clusters(rows: list[dict], threshold: float = NEAR_DUP_THRESHOLD) -> tuple[dict, list]:
    """Union-find over pairs with cosine >= threshold. Uses prompt text only (no outcomes)."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    text = [dedup_text(r) for r in rows]
    X = TfidfVectorizer(analyzer="char_wb", ngram_range=(4, 4), sublinear_tf=True, dtype=np.float32).fit_transform(text)
    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    pairs = []
    step = 1000
    for i in range(0, X.shape[0], step):
        S = (X[i:i + step] @ X.T).tocoo()
        keep = (S.data >= threshold) & (S.row + i < S.col)
        for a, b, s in zip(S.row[keep] + i, S.col[keep], S.data[keep]):
            pairs.append((int(a), int(b), float(s)))
            ra, rb = find(int(a)), find(int(b))
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
    members = defaultdict(list)
    for i in range(len(rows)):
        members[find(i)].append(i)
    cluster_of = {}
    for idxs in members.values():
        cid = min(rows[i]["qid"] for i in idxs)
        for i in idxs:
            cluster_of[rows[i]["qid"]] = cid
    return cluster_of, pairs


def build_manifest(rows: list[dict]) -> tuple[list[dict], dict]:
    corpus = [r for r in rows if r["role"] != "fresh"]
    fresh = [r for r in rows if r["role"] == "fresh"]
    cluster_of, pairs = near_duplicate_clusters(corpus)
    by_qid = {r["qid"]: r for r in corpus}
    size = defaultdict(int)
    for q, c in cluster_of.items():
        size[c] += 1
    # a cluster is stratified under the benchmark of its representative member
    clusters_by_bench = defaultdict(list)
    for c in sorted(set(cluster_of.values())):
        clusters_by_bench[by_qid[c]["benchmark"]].append(c)

    split_of, fold_of = {}, {}
    for bench, cl in clusters_by_bench.items():
        order = sorted(cl, key=lambda c: _h(SALT, "split", c))
        n = len(order)
        dev = []
        for rank, c in enumerate(order):
            pos = (rank + 0.5) / n
            if pos < TEST_FRACTION:
                split_of[c] = "test"
            elif pos < TEST_FRACTION + VAL_FRACTION:
                split_of[c] = "val"
                dev.append(c)
            else:
                split_of[c] = "train"
                dev.append(c)
        for rank, c in enumerate(sorted(dev, key=lambda c: _h(SALT, "fold", c))):
            fold_of[c] = rank % N_FOLDS

    manifest = []
    for r in corpus:
        c = cluster_of[r["qid"]]
        manifest.append(dict(qid=r["qid"], benchmark=r["benchmark"], role=r["role"], cluster_id=c, cluster_size=size[c],
                             split=split_of[c], cv_fold=fold_of.get(c, ""), eligible=int(r["eligible"]),
                             eligible_recovered=int(r["eligible_recovered"])))
    for r in fresh:
        manifest.append(dict(qid=r["qid"], benchmark=r["benchmark"], role=r["role"], cluster_id=r["qid"], cluster_size=1,
                             split="fresh_sealed", cv_fold="", eligible=int(r["eligible"]),
                             eligible_recovered=int(r["eligible_recovered"])))
    manifest.sort(key=lambda m: m["qid"])
    info = dict(n_pairs_at_threshold=len(pairs), n_clusters=len(size),
                n_multi_member_clusters=sum(1 for v in size.values() if v > 1),
                rows_in_multi_member_clusters=sum(v for v in size.values() if v > 1),
                largest_cluster=max(size.values()))
    return manifest, info


def write_manifest(manifest: list[dict], extra_meta: dict) -> dict:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(manifest)
    counts = defaultdict(lambda: defaultdict(int))
    for m in manifest:
        if m["eligible"]:
            counts[m["benchmark"]][m["split"]] += 1
    meta = dict(manifest_sha256=D.sha256_file(MANIFEST_PATH), paired_sha256=D.sha256_file(D.PAIRED_PATH), salt=SALT,
                test_fraction=TEST_FRACTION, val_fraction=VAL_FRACTION, n_folds=N_FOLDS,
                near_dup_threshold=NEAR_DUP_THRESHOLD, n_rows=len(manifest),
                eligible_counts={b: dict(v) for b, v in counts.items()}, **extra_meta)
    META_PATH.write_text(json.dumps(meta, indent=2))
    return meta


def read_manifest() -> dict[str, dict]:
    meta = json.loads(META_PATH.read_text())
    if D.sha256_file(MANIFEST_PATH) != meta["manifest_sha256"]:
        raise RuntimeError("split manifest does not match its recorded hash")
    if D.sha256_file(D.PAIRED_PATH) != meta["paired_sha256"]:
        raise RuntimeError("paired data does not match the hash recorded in the split manifest")
    with open(MANIFEST_PATH) as f:
        return {m["qid"]: m for m in csv.DictReader(f)}


def _log_access(purpose: str, what: str, n: int) -> None:
    ACCESS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(ACCESS_LOG, "a") as f:
        f.write(f"{_dt.datetime.now().isoformat(timespec='seconds')}\t{purpose}\t{what}\t{n}\n")


# Team decision 2026-11-16: the source pool for router development (algorithm selection,
# training, validation, cutoff tuning) and for the transfer study is SimpleQA, GPQA and
# MMLU-Pro without its math and computer-science questions, minus any question that
# duplicates a held-out benchmark question (review round, same day).
SOURCE_POOL_EXCLUDED_MMLU_SUBJECTS = ("math", "computer science")


HELD_OUT_ROLES = ("holdout_hle", "holdout_code", "holdout_math")


@functools.lru_cache(maxsize=1)
def boundary_duplicate_qids() -> frozenset:
    """Source-pool questions whose near-duplicate group also contains a question from a
    held-out benchmark (HLE, LiveCodeBench, AIME, LiveMathBench). They are removed from the
    source pool so that nothing the transfer router trains on is a copy of a target question."""
    man = read_manifest()
    held = {m["cluster_id"] for m in man.values() if m["role"] in HELD_OUT_ROLES}
    return frozenset(q for q, m in man.items() if m["role"] == "selection_pool" and m["cluster_id"] in held)


def in_source_pool(r: dict) -> bool:
    return (r["role"] == "selection_pool"
            and not (r["benchmark"] == "mmlupro" and r["subject"] in SOURCE_POOL_EXCLUDED_MMLU_SUBJECTS)
            and r["qid"] not in boundary_duplicate_qids())


def load_selection_dev(purpose: str, filtered: bool = True) -> tuple[D.Table, np.ndarray]:
    """Development rows (train+val) of the source pool with their CV fold. This is the ONLY
    outcome data algorithm selection may read.

    filtered=True (current protocol): SimpleQA, GPQA, MMLU-Pro minus math and computer science.
    filtered=False reproduces the first selection run of 2026-11-15 (all of MMLU-Pro)."""
    man = read_manifest()
    rows = [r for r in D.read_paired()
            if r["eligible"] and r["role"] == "selection_pool" and man[r["qid"]]["split"] in ("train", "val")
            and (in_source_pool(r) or not filtered)]
    rows.sort(key=lambda r: r["qid"])
    folds = np.array([int(man[r["qid"]]["cv_fold"]) for r in rows])
    _log_access(purpose, "source_pool_dev_filtered" if filtered else "selection_pool_dev_unfiltered", len(rows))
    return D.to_table(rows), folds


def _select(all_rows, man, splits, benchmarks, use_recovered_cost, exclude):
    key = "eligible_recovered" if use_recovered_cost else "eligible"
    rows = [r for r in all_rows if r[key] and man[r["qid"]]["split"] in splits
            and (benchmarks is None or r["benchmark"] in benchmarks) and not (exclude and exclude(r))]
    rows.sort(key=lambda r: r["qid"])
    return rows


def plan_counts(*, splits: tuple[str, ...], benchmarks: tuple[str, ...] | None = None,
                use_recovered_cost: bool = False, exclude: "callable | None" = None) -> dict:
    """Row counts per benchmark for a selection, using exactly the filter load_rows uses.
    Returns counts only: no outcome leaves this function, so it needs no unlock."""
    from collections import Counter
    rows = _select(D.read_paired(), read_manifest(), splits, benchmarks, use_recovered_cost, exclude)
    return dict(sorted(Counter(r["benchmark"] for r in rows).items()))


def load_rows(purpose: str, *, splits: tuple[str, ...], benchmarks: tuple[str, ...] | None = None,
              unlock_final: bool = False, use_recovered_cost: bool = False,
              exclude: "callable | None" = None) -> tuple[D.Table, list[dict]]:
    """General loader. Reading holdout benchmarks, the locked test split, or the fresh
    sealed sets requires unlock_final=True; every call is written to logs/data_access.log."""
    man = read_manifest()
    rows = _select(D.read_paired(), man, splits, benchmarks, use_recovered_cost, exclude)
    sealed = [r for r in rows if man[r["qid"]]["split"] in ("test", "fresh_sealed") or r["role"] != "selection_pool"]
    if sealed and not unlock_final:
        raise PermissionError(
            f"{len(sealed)} requested rows are sealed (locked test split, reserved holdout benchmark, or fresh set). "
            "They may only be read by the final-evaluation scripts after the protocol is frozen (unlock_final=True).")
    what = f"splits={','.join(splits)};benchmarks={','.join(benchmarks) if benchmarks else 'all'};unlock_final={unlock_final}"
    _log_access(purpose, what, len(rows))
    return D.to_table(rows, use_recovered_cost=use_recovered_cost), rows
