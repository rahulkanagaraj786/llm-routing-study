"""Token-length audit for ModernBERT. Reads prompt text only (no outcomes)."""
import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from router import data as D
from transformers import AutoTokenizer
from huggingface_hub import model_info

MODEL = "answerdotai/ModernBERT-base"
info = model_info(MODEL)
print("model", MODEL, "revision", info.sha)
tok = AutoTokenizer.from_pretrained(MODEL, revision=info.sha)
rows = [r for r in D.read_paired() if r["eligible_recovered"]]
out = dict(model=MODEL, revision=info.sha, per_benchmark={})
print(f"{'benchmark':14s} {'n':>5s} {'median':>7s} {'p90':>7s} {'p99':>7s} {'max':>7s} {'>512':>7s} {'>1024':>7s} {'>2048':>7s}")
for b in D.BENCHMARKS:
    P = [r["prompt"] for r in rows if r["benchmark"] == b]
    n = np.array([len(x) for x in tok(P, add_special_tokens=True, truncation=False)["input_ids"]])
    st = dict(n=len(P), median=float(np.median(n)), p90=float(np.quantile(n, .9)), p99=float(np.quantile(n, .99)), max=int(n.max()),
              frac_gt_512=float((n > 512).mean()), frac_gt_1024=float((n > 1024).mean()), frac_gt_2048=float((n > 2048).mean()))
    out["per_benchmark"][b] = st
    print(f"{b:14s} {st['n']:5d} {st['median']:7.0f} {st['p90']:7.0f} {st['p99']:7.0f} {st['max']:7d} {st['frac_gt_512']:7.1%} {st['frac_gt_1024']:7.1%} {st['frac_gt_2048']:7.1%}")
(D.ROOT / "results/audit").mkdir(parents=True, exist_ok=True)
(D.ROOT / "results/audit/token_lengths.json").write_text(json.dumps(out, indent=2))
