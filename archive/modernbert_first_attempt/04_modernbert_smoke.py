"""Timing-only smoke test of the ModernBERT trainer on this machine.

Uses selection-pool development prompts so that batch lengths are realistic. It runs a
few optimisation steps and reports speed and memory. It deliberately does NOT compute
or print any routing or accuracy metric.

On Apple MPS, run with PYTORCH_MPS_HIGH_WATERMARK_RATIO set (e.g. 0.6) so that the
process fails with an out-of-memory error instead of pushing the machine into swap."""
import sys, time, json, platform, os
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from router import splits as S, modernbert as MB
import torch

tab, folds = S.load_selection_dev("modernbert_timing_smoke_test")
n_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 30
bs = int(sys.argv[2]) if len(sys.argv) > 2 else 16
fit = np.flatnonzero(np.isin(folds, [2, 3, 4]))
cal = np.flatnonzero(folds == 1)[:32]
r = MB.ModernBertRouter(lr=3e-5, batch_size=bs, max_epochs=1)
t0 = time.perf_counter()
r._load(); print("model loaded on", r.device, f"in {time.perf_counter()-t0:.1f}s", flush=True)
ids, _ = r._encode(tab.prompts[fit]); labels = np.array([MB.LABEL_MAP[int(v)] for v in tab.outcomes.y[fit]])
opt = torch.optim.AdamW(r.model.parameters(), lr=3e-5)
rng = np.random.default_rng(0); r.model.train(); times = []; lens = []
for bi, idx in enumerate(r._batches(ids, bs, True, rng)):
    ts = time.perf_counter()
    x, a = r._collate(ids, idx)
    loss = torch.nn.functional.cross_entropy(r.model(input_ids=x, attention_mask=a).logits, torch.tensor(labels[idx], device=r.device))
    loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
    float(loss)  # force synchronisation
    times.append(time.perf_counter() - ts); lens.append(int(x.shape[1]))
    mem = torch.mps.driver_allocated_memory() / 2**30 if r.device.type == "mps" else float("nan")
    print(f"step {bi+1:3d} padded_len {x.shape[1]:5d} {times[-1]:6.2f}s  mps_mem {mem:.2f} GB", flush=True)
    if bi + 1 >= n_steps:
        break
steps_per_epoch = int(np.ceil(len(fit) / bs))
per_step = float(np.median(times[2:])) if len(times) > 4 else float(np.mean(times))
t1 = time.perf_counter(); g = r.predict_gain(tab.prompts[cal]); pred_s = (time.perf_counter() - t1) / len(cal)
out = dict(device=str(r.device), machine=platform.machine(), torch=torch.__version__, batch_size=bs, steps_timed=len(times),
           median_seconds_per_step=per_step, mean_seconds_per_step=float(np.mean(times)), padded_lengths=lens,
           train_rows_per_fit=int(len(fit)), steps_per_epoch=steps_per_epoch, est_minutes_per_epoch=per_step * steps_per_epoch / 60,
           predict_seconds_per_prompt=pred_s, mps_driver_memory_gb=mem, finite_predictions=bool(np.isfinite(g).all()),
           watermark=os.environ.get("PYTORCH_MPS_HIGH_WATERMARK_RATIO"), total_seconds=time.perf_counter() - t0)
print(json.dumps({k: v for k, v in out.items() if k != "padded_lengths"}, indent=1))
(S.D.ROOT / "results/audit").mkdir(parents=True, exist_ok=True)
(S.D.ROOT / f"results/audit/modernbert_smoke_{r.device.type}_bs{bs}.json").write_text(json.dumps(out, indent=1))
