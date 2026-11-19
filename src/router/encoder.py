"""Fine-tuned text encoder: ModernBERT-base with a three-class classification head.

The whole network (encoder and head) is fine-tuned with cross-entropy on Y + 1 in {0, 1, 2};
gain = P(Y = +1) - P(Y = -1). Input is the prompt text only.

Training recipe (every setting is a constructor argument and is recorded with the results):
  - answerdotai/ModernBERT-base at a pinned revision; prompts truncated to 1,024 tokens
  - AdamW, weight decay 0.01 (none on biases and norms), 10% linear warm-up then linear decay,
    gradient clipping at 1.0
  - batches of 16 prompts of similar length (length-grouped, batch order shuffled). A batch is
    computed in micro-batches of at most 4,096 padded tokens with gradients accumulated, so the
    update is that of the full batch of 16 while memory stays within a 16 GB GPU
  - up to 3 epochs; the validation questions passed to fit() are scored 4 times per epoch, the
    best snapshot is kept, and training stops after 3 checks in a row without improvement
  - mixed precision on CUDA: bf16 on GPUs that support it natively (A100, L4, H100), fp16 with
    loss scaling on older ones (T4); fp32 on Apple MPS and CPU. Seed 221.

Device: CUDA if available, then Apple MPS, then CPU. A micro-batch that still runs out of GPU
memory in its forward pass is split in half and retried (the update is the same), and the split
is counted in `oom_splits`.
"""
from __future__ import annotations

import contextlib
import gc
import json
import time

import numpy as np

from .base import Router

SEED = 221
MODEL_NAME = "answerdotai/ModernBERT-base"
MODEL_REVISION = "8949b909ec900327062f0ebf497f51aef5e6f0c8"


def _device():
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _is_oom(err: BaseException) -> bool:
    import torch
    return isinstance(err, torch.OutOfMemoryError) or "out of memory" in str(err).lower()


def _native_bf16() -> bool:
    """True only on GPUs with real bf16 support (Ampere or newer: A100, L4, H100...). On older
    cards such as the T4, torch reports bf16 as available but emulates it slowly."""
    import torch
    try:
        return bool(torch.cuda.is_bf16_supported(including_emulation=False))
    except TypeError:  # older torch without the argument
        return torch.cuda.get_device_capability()[0] >= 8


def length_grouped_batches(lengths, batch_size: int, rng, mega: int = 50) -> list[np.ndarray]:
    """Shuffle, cut into chunks of `mega` batches, sort each chunk by length, split it into
    batches, then shuffle the batch order. Batches hold prompts of similar length, so little
    compute goes to padding, while which prompts meet and the batch order stay random.
    (The same idea as `group_by_length` in Hugging Face's Trainer.)"""
    perm = rng.permutation(len(lengths))
    lengths = np.asarray(lengths)
    batches = []
    step = batch_size * mega
    for s in range(0, len(perm), step):
        chunk = perm[s: s + step]
        chunk = chunk[np.argsort(-lengths[chunk], kind="stable")]
        batches += [chunk[i: i + batch_size] for i in range(0, len(chunk), batch_size)]
    return [batches[i] for i in rng.permutation(len(batches))]


def token_chunks(lengths_desc, budget: int, cap: int | None = None) -> list[list[int]]:
    """Split positions 0..n-1 (lengths sorted longest first) into consecutive groups whose padded
    size, count x longest length, stays within `budget` tokens (at least one prompt per group)."""
    parts, cur = [], []
    for j, L in enumerate(lengths_desc):
        longest = lengths_desc[cur[0]] if cur else L
        if cur and ((len(cur) + 1) * longest > budget or (cap and len(cur) >= cap)):
            parts.append(cur)
            cur = []
        cur.append(j)
    if cur:
        parts.append(cur)
    return parts


def check_points(steps_per_epoch: int, checks_per_epoch: int) -> set[int]:
    """Batch numbers (1-based, within an epoch) after which the validation questions are scored.
    The last batch of the epoch is always one of them."""
    k = max(1, min(checks_per_epoch, steps_per_epoch))
    return {int(round(steps_per_epoch * j / k)) for j in range(1, k + 1)}


_FITS_STARTED = 0  # fine-tunes started in this Python process, for the progress lines


def _minutes(seconds: float) -> str:
    return f"{seconds / 60:.1f} min" if seconds >= 60 else f"{seconds:.0f} s"


def _bar(frac: float, width: int = 20) -> str:
    n = int(round(max(0.0, min(1.0, frac)) * width))
    return "[" + "#" * n + "." * (width - n) + "]"


def _empty_cache(device) -> None:
    import torch
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


class EncoderRouter(Router):
    family = "encoder"

    def __init__(self, lr: float = 2e-5, max_length: int = 1024, batch_size: int = 16, max_epochs: int = 3,
                 checks_per_epoch: int = 4, patience: int = 3, weight_decay: float = 0.01, warmup_frac: float = 0.1,
                 max_grad_norm: float = 1.0, mixed_precision: bool = True, group_by_length: bool = True,
                 max_batch_tokens: int = 4096, model_name: str = MODEL_NAME, revision: str = MODEL_REVISION):
        super().__init__(lr=lr, max_length=max_length, batch_size=batch_size, max_epochs=max_epochs,
                         checks_per_epoch=checks_per_epoch, patience=patience, weight_decay=weight_decay,
                         warmup_frac=warmup_frac, max_grad_norm=max_grad_norm, mixed_precision=mixed_precision,
                         group_by_length=group_by_length, max_batch_tokens=max_batch_tokens,
                         model_name=model_name, revision=revision)
        self.model = None
        self.tokenizer = None
        self.device = None
        self.scaler = None
        self._in_backward = False
        self.history: list[dict] = []
        self.best_epoch = None
        self.oom_splits = 0
        self.truncation_rate_fit = None

    # ---- precision ----------------------------------------------------------------------
    @property
    def precision(self) -> str:
        if self.device is None or self.device.type != "cuda" or not self.config["mixed_precision"]:
            return "fp32"
        return "bf16-mixed" if _native_bf16() else "fp16-mixed"

    def _autocast(self):
        import torch
        dtype = {"bf16-mixed": torch.bfloat16, "fp16-mixed": torch.float16}.get(self.precision)
        return torch.autocast(self.device.type, dtype=dtype) if dtype is not None else contextlib.nullcontext()

    # ---- helpers ------------------------------------------------------------------------
    def _encode(self, prompts) -> list[list[int]]:
        enc = self.tokenizer([str(p) for p in prompts], truncation=True, max_length=self.config["max_length"],
                             add_special_tokens=True)
        return enc["input_ids"]

    def _collate(self, ids: list[list[int]]):
        import torch
        n, width = len(ids), max(len(x) for x in ids)
        input_ids = torch.full((n, width), self.tokenizer.pad_token_id, dtype=torch.long)
        mask = torch.zeros((n, width), dtype=torch.long)
        for i, x in enumerate(ids):
            input_ids[i, : len(x)] = torch.tensor(x, dtype=torch.long)
            mask[i, : len(x)] = 1
        return input_ids.to(self.device), mask.to(self.device)

    def _forward_backward(self, ids, labels, n_total) -> float:
        """Forward + backward on one micro-batch. The loss is divided by the size of the whole
        batch, so accumulating micro-batches gives exactly the gradient of the full batch."""
        import torch
        input_ids, mask = self._collate(ids)
        y = torch.tensor(labels, dtype=torch.long, device=self.device)
        with self._autocast():
            logits = self.model(input_ids=input_ids, attention_mask=mask).logits
        loss = torch.nn.functional.cross_entropy(logits.float(), y, reduction="sum")
        self._in_backward = True
        self.scaler.scale(loss / n_total).backward()
        self._in_backward = False
        return float(loss.detach())

    def _train_batch(self, ids, labels) -> float:
        """One optimiser batch, run as micro-batches of at most `max_batch_tokens` padded tokens.
        Returns the summed loss."""
        order = sorted(range(len(ids)), key=lambda i: -len(ids[i]))
        queue = [[order[j] for j in part]
                 for part in token_chunks([len(ids[i]) for i in order], self.config["max_batch_tokens"])]
        total = 0.0
        while queue:
            part = queue.pop(0)
            oom = False
            try:
                total += self._forward_backward([ids[i] for i in part], [labels[i] for i in part], len(ids))
            except Exception as err:  # noqa: BLE001 - only out-of-memory is handled
                if not _is_oom(err) or len(part) == 1:
                    raise
                if self._in_backward:  # some gradients may already be added: the batch cannot be redone
                    raise RuntimeError("out of GPU memory during the backward pass; lower max_batch_tokens") from err
                oom = True
            if oom:  # retried outside the except block, so the failed attempt's memory is released first
                self.oom_splits += 1
                _empty_cache(self.device)
                h = len(part) // 2
                queue[:0] = [part[:h], part[h:]]
        return total

    def _gains(self, ids: list[list[int]]) -> np.ndarray:
        import torch
        was_training = self.model.training
        self.model.eval()
        order = sorted(range(len(ids)), key=lambda i: -len(ids[i]))  # similar lengths together: less padding
        budget = 4 * self.config["max_batch_tokens"]  # no gradients kept: larger batches fit
        queue = [[order[j] for j in part] for part in token_chunks([len(ids[i]) for i in order], budget, cap=128)]
        out = np.zeros(len(ids))
        with torch.no_grad():
            while queue:
                part = queue.pop(0)
                p, oom = None, False
                try:
                    input_ids, mask = self._collate([ids[j] for j in part])
                    with self._autocast():
                        logits = self.model(input_ids=input_ids, attention_mask=mask).logits
                    p = torch.softmax(logits.float(), dim=-1).cpu().numpy()
                except Exception as err:  # noqa: BLE001
                    if not _is_oom(err) or len(part) == 1:
                        raise
                    oom = True
                if oom:
                    _empty_cache(self.device)
                    h = len(part) // 2
                    queue[:0] = [part[:h], part[h:]]
                    continue
                out[part] = p[:, 2] - p[:, 0]
        if was_training:
            self.model.train()
        return np.clip(out, -1.0, 1.0)

    # ---- contract -----------------------------------------------------------------------
    def fit(self, prompts, y, val_prompts=None, val_y=None):
        import warnings

        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from transformers.utils import logging as hf_logging

        global _FITS_STARTED
        _FITS_STARTED += 1
        fit_no = _FITS_STARTED
        hf_logging.set_verbosity_error()  # hides the expected "classifier newly initialised" load report
        # with fp16 loss scaling the first updates can be skipped while the scale settles; harmless
        warnings.filterwarnings("ignore", message="Detected call of `lr_scheduler.step\\(\\)` before")

        t0 = time.perf_counter()
        c = self.config
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        self.device = _device()
        # loss scaling is only needed for fp16; for bf16 / fp32 the scaler is a no-op
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=self.precision == "fp16-mixed")

        self.tokenizer = AutoTokenizer.from_pretrained(c["model_name"], revision=c["revision"])
        self.model = AutoModelForSequenceClassification.from_pretrained(
            c["model_name"], revision=c["revision"], num_labels=3).to(self.device)

        ids = self._encode(prompts)
        full = self.tokenizer([str(p) for p in prompts], add_special_tokens=True)["input_ids"]
        self.truncation_rate_fit = float(np.mean([len(x) > c["max_length"] for x in full]))
        labels = (np.asarray(y, dtype=int) + 1).tolist()
        val_ids = self._encode(val_prompts) if val_prompts is not None and len(val_prompts) else None
        val_y = np.asarray(val_y, dtype=float) if val_ids is not None else None

        bs = c["batch_size"]
        lengths = [len(x) for x in ids]

        def epoch_batches(epoch):
            rng = np.random.default_rng(SEED + epoch)
            if c["group_by_length"]:
                return length_grouped_batches(lengths, bs, rng)
            perm = rng.permutation(len(ids))
            return [perm[i: i + bs] for i in range(0, len(perm), bs)]

        steps_per_epoch = len(epoch_batches(1))  # the same for every epoch
        total = steps_per_epoch * c["max_epochs"]
        warm = max(1, int(round(c["warmup_frac"] * total)))
        decay, no_decay = [], []
        for name, p in self.model.named_parameters():
            (no_decay if p.ndim < 2 or "norm" in name.lower() else decay).append(p)
        opt = torch.optim.AdamW([dict(params=decay, weight_decay=c["weight_decay"]),
                                 dict(params=no_decay, weight_decay=0.0)], lr=c["lr"])
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (total - s) / max(1, total - warm)))
        checks = check_points(steps_per_epoch, c["checks_per_epoch"]) if val_ids is not None else {steps_per_epoch}
        # progress line every ENCODER_PROGRESS_SECONDS (default 20 s). If the caller sets
        # ENCODER_TOTAL_FITS (the notebook does), the line also shows progress over the whole run.
        import os
        every_s = float(os.environ.get("ENCODER_PROGRESS_SECONDS", "20"))
        total_fits = int(os.environ.get("ENCODER_TOTAL_FITS", "0") or 0)
        last_report = time.perf_counter()
        print(f"  [fine-tune #{fit_no}{f' of {total_fits}' if total_fits else ''}] lr {c['lr']:g} | {len(ids):,} training questions, "
              f"{len(val_ids) if val_ids is not None else 0:,} validation | {steps_per_epoch} batches per epoch, "
              f"up to {c['max_epochs']} epochs | {self.device.type}, {self.precision}", flush=True)
        t_train = time.perf_counter()

        best_mse, best_state, bad, stop = np.inf, None, 0, False
        self.model.train()
        for epoch in range(1, c["max_epochs"] + 1):
            te = time.perf_counter()
            loss_sum, n_seen, padded = 0.0, 0, 0
            for b, idx in enumerate(epoch_batches(epoch), start=1):
                padded += len(idx) * max(lengths[i] for i in idx)
                opt.zero_grad(set_to_none=True)
                loss_sum += self._train_batch([ids[i] for i in idx], [labels[i] for i in idx])
                n_seen += len(idx)
                self.scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), c["max_grad_norm"])
                self.scaler.step(opt)
                self.scaler.update()
                sched.step()
                done = (epoch - 1) * steps_per_epoch + b
                now = time.perf_counter()
                if now - last_report >= every_s and b not in checks:
                    last_report = now
                    spent = now - t_train
                    frac = done / total
                    left = spent / done * (total - done)
                    line = (f"    {_bar(frac)} {100 * frac:3.0f}% of fine-tune | epoch {epoch}/{c['max_epochs']} "
                            f"batch {b}/{steps_per_epoch} | {_minutes(spent)} elapsed, at most ~{_minutes(left)} left")
                    if total_fits:
                        run = min(1.0, (fit_no - 1 + frac) / total_fits)
                        run_left = left + (total_fits - fit_no) * (spent / frac)
                        line += f" || whole run {_bar(run, 10)} {100 * run:3.0f}%, at most ~{_minutes(run_left)} left"
                    print(line, flush=True)
                if b not in checks:
                    continue
                rec = dict(epoch=round(epoch - 1 + b / steps_per_epoch, 3), step=(epoch - 1) * steps_per_epoch + b,
                           train_loss=loss_sum / n_seen, seconds=time.perf_counter() - te, oom_splits=self.oom_splits,
                           padding_fraction=1.0 - sum(lengths) / max(1, padded) if b == steps_per_epoch else None)
                if val_ids is not None:
                    mse = float(np.mean((self._gains(val_ids) - val_y) ** 2))
                    rec["val_advantage_mse"] = mse
                    improved = mse < best_mse - 1e-12
                    print(f"    epoch {rec['epoch']:.2f}: train loss {rec['train_loss']:.4f}  val advantage MSE "
                          f"{mse:.5f}{'  *best' if improved else ''}  ({rec['seconds']:.0f}s, {self.device.type}, "
                          f"{self.precision})", flush=True)
                    if improved:
                        best_mse, bad, self.best_epoch = mse, 0, rec["epoch"]
                        best_state = {k: v.detach().to("cpu", copy=True) for k, v in self.model.state_dict().items()}
                    else:
                        bad += 1
                        stop = bad >= c["patience"]
                self.history.append(rec)
                if stop:
                    break
            if stop:
                break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        else:
            self.best_epoch = self.history[-1]["epoch"] if self.history else None
        self.model.eval()
        self.fit_seconds = time.perf_counter() - t0
        last = self.history[-1]["epoch"] if self.history else 0
        stopped = f"stopped early at epoch {last:.2f}" if last < c["max_epochs"] else "ran all epochs"
        print(f"  [fine-tune #{fit_no}{f' of {total_fits}' if total_fits else ''}] done in {_minutes(self.fit_seconds)}: {stopped}, best at epoch "
              f"{self.best_epoch}" + (f" (val advantage MSE {best_mse:.5f})" if best_state is not None else ""),
              flush=True)
        return self

    def predict_gain(self, prompts) -> np.ndarray:
        return self._gains(self._encode(prompts)).astype(float)

    # ---- optional hooks -----------------------------------------------------------------
    def extra_info(self) -> dict:
        import torch
        dev = self.device
        gpu = torch.cuda.get_device_name(0) if dev is not None and dev.type == "cuda" else None
        return dict(history=self.history, best_epoch=self.best_epoch, device=dev.type if dev else None, gpu=gpu,
                    precision=self.precision, truncation_rate_fit=self.truncation_rate_fit,
                    oom_splits=self.oom_splits, full_config=dict(self.config))

    def save(self, path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        (path / "router_info.json").write_text(json.dumps(self.extra_info(), indent=2))

    def release(self) -> None:
        dev = self.device
        self.model = None
        if dev is not None:
            _empty_cache(dev)


# Learning rates to try: the usual low / high pair for fine-tuning BERT-style models.
ENCODER_GRID = [dict(lr=2e-5), dict(lr=5e-5)]
