"""Shared data layer: build the paired GPT-5 / Gemini 2.5 Flash table from the raw
LLMRouterBench release, and load it for experiments.

Design rule enforced here: predictors only ever receive `prompts` (text available
before generation). Outcomes (scores, costs) live in a separate `Outcomes` object
that is handed to the label builder and to the policy evaluator, never to a
featurizer or a predictor's `predict`.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / "extracted" / "bench-release"
PAIRED_PATH = ROOT / "data" / "processed" / "paired_v2.jsonl.gz"

STRONG = "gpt-5"
CHEAP = "gemini-2.5-flash"

# List prices implied by the recorded costs (verified in the audit: cost equals
# prompt_tokens * in + completion_tokens * out to ~1e-14 relative error on most files).
IMPLIED_PRICE_PER_TOKEN = {
    STRONG: (1.25e-6, 10.0e-6),
    CHEAP: (0.30e-6, 2.50e-6),
}

# benchmark -> raw location, coarse domain and role in the study design.
#   selection_pool : used for algorithm selection and tuning (dev portion only)
#   holdout_*      : reserved from algorithm selection/tuning; RQ2 targets
#   fresh          : never part of the pilot corpus; sealed until final evaluation
BENCHMARKS = {
    "simpleqa": dict(dataset="simpleqa", split="test", domain="knowledge", role="selection_pool"),
    "mmlupro": dict(dataset="mmlupro", split="test_3000", domain="knowledge", role="selection_pool"),
    "gpqa": dict(dataset="gpqa", split="test", domain="knowledge", role="selection_pool"),
    "hle": dict(dataset="hle", split="test", domain="knowledge", role="holdout_hle"),
    "livecodebench": dict(dataset="livecodebench", split="test", domain="code", role="holdout_code"),
    "aime": dict(dataset="aime", split="hybrid", domain="math", role="holdout_math"),
    "livemathbench": dict(dataset="livemathbench", split="test", domain="math", role="holdout_math"),
    "arc-agi": dict(dataset="arc-agi", split="v1", domain="abstract_reasoning", role="fresh"),
    "swe-bench": dict(dataset="swe-bench", split="verified", domain="code", role="fresh"),
}
CORPUS_BENCHMARKS = [b for b, v in BENCHMARKS.items() if v["role"] != "fresh"]
SELECTION_POOL = [b for b, v in BENCHMARKS.items() if v["role"] == "selection_pool"]

_MMLU_SUBJECT = re.compile(r"^Answer the following (.+?) question\.")
_WS = re.compile(r"[\s\u2028\u2029\u00a0]+")


def _norm_ws(s: str) -> str:
    """Whitespace-insensitive form used only to decide whether two prompts are the same question."""
    return _WS.sub(" ", s).strip()


def list_price_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float | None:
    """Cost at the price schedule implied by the release. Sensitivity analysis only."""
    if prompt_tokens is None or completion_tokens is None or prompt_tokens <= 0 or completion_tokens <= 0:
        return None
    pin, pout = IMPLIED_PRICE_PER_TOKEN[model]
    return prompt_tokens * pin + completion_tokens * pout


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_gz_content(path: Path) -> str:
    """SHA-256 of the uncompressed bytes of a .gz file. Unlike the hash of the file itself,
    this is the same on every machine, whatever zlib build compressed it."""
    h = hashlib.sha256()
    with gzip.open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _raw_file(dataset: str, split: str, model: str) -> Path:
    files = sorted((RAW_DIR / dataset / split / model).glob("*.json"))
    if len(files) != 1:
        raise FileNotFoundError(f"expected exactly one result file for {dataset}/{split}/{model}, got {files}")
    return files[0]


def _qid(benchmark: str, prompt: str) -> str:
    return hashlib.sha256(f"{benchmark}\n{prompt}".encode("utf-8")).hexdigest()[:24]


def build_paired() -> tuple[list[dict], dict]:
    """Join GPT-5 and Flash records question by question. Returns rows and provenance."""
    rows: list[dict] = []
    prov: dict = {"files": [], "benchmarks": {}}

    # Sister file used only to look up recorded GPT-5 costs that are missing (0.0)
    # in hle/test although the identical response is costed in hle/subset_500.
    sister = {r["prompt"]: r for r in json.load(open(_raw_file("hle", "subset_500", STRONG)))["records"]}
    prov["files"].append(_file_prov(_raw_file("hle", "subset_500", STRONG), "hle", "subset_500", STRONG, None))

    for bench, meta in BENCHMARKS.items():
        fs, fc = _raw_file(meta["dataset"], meta["split"], STRONG), _raw_file(meta["dataset"], meta["split"], CHEAP)
        ds, dc = json.load(open(fs)), json.load(open(fc))
        prov["files"].append(_file_prov(fs, meta["dataset"], meta["split"], STRONG, ds))
        prov["files"].append(_file_prov(fc, meta["dataset"], meta["split"], CHEAP, dc))
        assert ds["model_name"] == STRONG and dc["model_name"] == CHEAP, (ds["model_name"], dc["model_name"])
        rs, rc = ds["records"], dc["records"]
        assert len(rs) == len(rc), (bench, len(rs), len(rc))
        prov["benchmarks"][bench] = dict(
            n_records=len(rs),
            fingerprint_strong=ds.get("data_fingerprint"),
            fingerprint_cheap=dc.get("data_fingerprint"),
            fingerprint_match=ds.get("data_fingerprint") == dc.get("data_fingerprint"),
        )
        for pos, (s, c) in enumerate(zip(rs, rc)):
            prompt = s["prompt"] if isinstance(s["prompt"], str) else json.dumps(s["prompt"], ensure_ascii=False)
            cprompt = c["prompt"] if isinstance(c["prompt"], str) else json.dumps(c["prompt"], ensure_ascii=False)
            subject = None
            if bench == "mmlupro":
                m = _MMLU_SUBJECT.match(prompt)
                subject = m.group(1) if m else None
            s_cost, c_cost = float(s["cost"] or 0.0), float(c["cost"] or 0.0)
            # recorded cost for the identical GPT-5 response in the sister file (HLE only)
            s_cost_sister = None
            if bench == "hle" and s_cost <= 0 and prompt in sister:
                z = sister[prompt]
                if z["raw_output"] == s["raw_output"] and z["completion_tokens"] == s["completion_tokens"] and z["cost"] > 0:
                    s_cost_sister = float(z["cost"])
            row = dict(
                qid=_qid(bench, prompt),
                benchmark=bench,
                domain=meta["domain"],
                role=meta["role"],
                subject=subject,
                position=pos,
                raw_index=s.get("index"),
                query=s.get("origin_query") if isinstance(s.get("origin_query"), str) else "",
                prompt=prompt,
                strong_score=s["score"],
                cheap_score=c["score"],
                strong_cost=s_cost,
                cheap_cost=c_cost,
                strong_cost_sister=s_cost_sister,
                strong_cost_list=list_price_cost(STRONG, s["prompt_tokens"], s["completion_tokens"]),
                cheap_cost_list=list_price_cost(CHEAP, c["prompt_tokens"], c["completion_tokens"]),
                strong_prompt_tokens=s["prompt_tokens"],
                cheap_prompt_tokens=c["prompt_tokens"],
                strong_completion_tokens=s["completion_tokens"],
                cheap_completion_tokens=c["completion_tokens"],
                same_prompt=_norm_ws(prompt) == _norm_ws(cprompt),
                same_prompt_exact=prompt == cprompt,
                same_index=s.get("index") == c.get("index"),
                same_ground_truth=s.get("ground_truth") == c.get("ground_truth"),
                strong_nonempty=bool(s.get("raw_output")),
                cheap_nonempty=bool(c.get("raw_output")),
            )
            row.update(eligibility(row))
            rows.append(row)
    return rows, prov


def _file_prov(path: Path, dataset: str, split: str, model: str, d: dict | None) -> dict:
    out = dict(path=str(path.relative_to(RAW_DIR.parent)), dataset=dataset, split=split, model_dir=model,
               bytes=path.stat().st_size, sha256=sha256_file(path))
    if d is not None:
        out.update(model_name=d.get("model_name"), counts=d.get("counts"), performance=d.get("performance"),
                   data_fingerprint=d.get("data_fingerprint"))
    return out


def eligibility(r: dict) -> dict:
    """Primary eligibility = the pilot's filter, reproduced from raw fields.

    `eligible_recovered` additionally admits HLE questions whose GPT-5 cost is 0.0
    in hle/test but recorded (>0) for the identical response in hle/subset_500.
    Whether to use it is an open team decision; it never affects the selection pool.
    """
    reasons = []
    if not r["same_prompt"]:
        reasons.append("prompt_mismatch")
    if r["strong_score"] not in (0, 1) or r["cheap_score"] not in (0, 1):
        reasons.append("non_binary_score")
    if not r["strong_nonempty"]:
        reasons.append("strong_empty_response")
    if not r["cheap_nonempty"]:
        reasons.append("cheap_empty_response")
    cost_reasons = []
    if not r["strong_cost"] > 0:
        cost_reasons.append("strong_cost_missing")
    if not r["cheap_cost"] > 0:
        cost_reasons.append("cheap_cost_missing")
    eligible = not reasons and not cost_reasons
    recoverable = (not reasons and cost_reasons == ["strong_cost_missing"] and r["strong_cost_sister"] is not None)
    return dict(eligible=eligible, eligible_recovered=eligible or recoverable,
                exclusion_reasons=reasons + cost_reasons)


def write_paired(rows: list[dict], path: Path = PAIRED_PATH) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    # mtime=0 and sorted keys so the file hash is reproducible
    with open(path, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
        for r in rows:
            gz.write((json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8"))
    return sha256_file(path)


def read_paired(path: Path = PAIRED_PATH) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


@dataclass(frozen=True)
class Outcomes:
    """Recorded outcomes. Never passed to a featurizer or predictor."""
    strong_correct: np.ndarray
    cheap_correct: np.ndarray
    strong_cost: np.ndarray
    cheap_cost: np.ndarray

    @property
    def y(self) -> np.ndarray:
        """Training target Y = GPT-5 correctness - Flash correctness, in {-1, 0, +1}."""
        return (self.strong_correct - self.cheap_correct).astype(int)

    def take(self, idx) -> "Outcomes":
        return Outcomes(self.strong_correct[idx], self.cheap_correct[idx], self.strong_cost[idx], self.cheap_cost[idx])

    def __len__(self) -> int:
        return len(self.strong_correct)


@dataclass(frozen=True)
class Table:
    qid: np.ndarray
    benchmark: np.ndarray
    prompts: np.ndarray  # object array of str: the only predictor input
    outcomes: Outcomes

    def take(self, idx) -> "Table":
        return Table(self.qid[idx], self.benchmark[idx], self.prompts[idx], self.outcomes.take(idx))

    def __len__(self) -> int:
        return len(self.qid)


def to_table(rows: list[dict], use_recovered_cost: bool = False) -> Table:
    def scost(r):
        if r["strong_cost"] > 0:
            return r["strong_cost"]
        if use_recovered_cost and r["strong_cost_sister"] is not None:
            return r["strong_cost_sister"]
        raise ValueError(f"row {r['qid']} has no usable GPT-5 cost")

    return Table(
        qid=np.array([r["qid"] for r in rows]),
        benchmark=np.array([r["benchmark"] for r in rows]),
        prompts=np.array([r["prompt"] for r in rows], dtype=object),
        outcomes=Outcomes(
            strong_correct=np.array([int(r["strong_score"]) for r in rows]),
            cheap_correct=np.array([int(r["cheap_score"]) for r in rows]),
            strong_cost=np.array([scost(r) for r in rows], dtype=float),
            cheap_cost=np.array([r["cheap_cost"] for r in rows], dtype=float),
        ),
    )
