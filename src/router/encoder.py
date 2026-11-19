"""Fine-tuned text encoder: ModernBERT-base with a three-class classification head.

The whole network (encoder and head) is fine-tuned with cross-entropy on Y + 1 in {0, 1, 2};
gain = P(Y = +1) - P(Y = -1). Input is the prompt text only.

Settings fixed in advance (docs/PROTOCOL.md, section 5). Only the learning rate is searched;
everything else is a constructor default and is recorded with every result:
  - answerdotai/ModernBERT-base at a pinned revision; prompts truncated to 1,024 tokens
  - AdamW, weight decay 0.01, 10% linear warm-up then linear decay, gradient clipping at 1.0
  - batches of 16 prompts of similar length (length-grouped, batch order shuffled), padded to
    the longest prompt in the batch
  - at most 3 epochs; early stopping on the validation questions passed to fit() (patience 1),
    keeping the weights of the best epoch
  - bf16 mixed precision on a CUDA GPU with native bf16 (A100, L4, ...), fp32 elsewhere
    (including the T4, which only emulates bf16); seed 221

Device: CUDA if available, then Apple MPS, then CPU. A batch that runs out of GPU memory is
split in half (gradients are accumulated, so the update is the same) and the split is counted.

Run on the GPU server: see docs/TEAM_GUIDE.md and the encoder notes in docs/PROTOCOL.md.
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
    cards such as the T4, torch reports bf16 as available but emulates it slowly; use fp32 there."""
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
                 patience: int = 1, weight_decay: float = 0.01, warmup_frac: float = 0.1, max_grad_norm: float = 1.0,
                 bf16: bool = True, group_by_length: bool = True, model_name: str = MODEL_NAME,
                 revision: str = MODEL_REVISION):
        super().__init__(lr=lr, max_length=max_length, batch_size=batch_size, max_epochs=max_epochs, patience=patience,
                         weight_decay=weight_decay, warmup_frac=warmup_frac, max_grad_norm=max_grad_norm, bf16=bf16,
                         group_by_length=group_by_length, model_name=model_name, revision=revision)
        self.model = None
        self.tokenizer = None
        self.history: list[dict] = []
        self.best_epoch = None
        self.oom_splits = 0
        self.truncation_rate_fit = None

    # ---- helpers ------------------------------------------------------------------------
    def _autocast(self):
        import torch
        if self.precision == "bf16-mixed":
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    @property
    def precision(self) -> str:
        if self.device.type == "cuda" and self.config["bf16"] and _native_bf16():
            return "bf16-mixed"
        return "fp32"

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

    def _train_step(self, ids, labels, n_total) -> float:
        """Forward + backward on one batch; halves it on out-of-memory. Returns summed loss."""
        import torch
        try:
            input_ids, mask = self._collate(ids)
            y = torch.tensor(labels, dtype=torch.long, device=self.device)
            with self._autocast():
                logits = self.model(input_ids=input_ids, attention_mask=mask).logits
            loss = torch.nn.functional.cross_entropy(logits.float(), y, reduction="sum")
            (loss / n_total).backward()
            return float(loss.detach())
        except Exception as err:  # noqa: BLE001 - only out-of-memory is handled
            if not _is_oom(err) or len(ids) == 1:
                raise
            self.oom_splits += 1
            _empty_cache(self.device)
            h = len(ids) // 2
            return (self._train_step(ids[:h], labels[:h], n_total)
                    + self._train_step(ids[h:], labels[h:], n_total))

    def _gains(self, ids: list[list[int]], batch_size: int = 64) -> np.ndarray:
        import torch
        self.model.eval()
        order = np.argsort([len(x) for x in ids], kind="stable")  # similar lengths together: less padding
        out = np.zeros(len(ids))
        with torch.no_grad():
            i = 0
            while i < len(order):
                idx = order[i: i + batch_size]
                try:
                    input_ids, mask = self._collate([ids[j] for j in idx])
                    with self._autocast():
                        logits = self.model(input_ids=input_ids, attention_mask=mask).logits
                except Exception as err:  # noqa: BLE001
                    if not _is_oom(err) or batch_size == 1:
                        raise
                    _empty_cache(self.device)
                    batch_size = max(1, batch_size // 2)
                    continue
                p = torch.softmax(logits.float(), dim=-1).cpu().numpy()
                out[idx] = p[:, 2] - p[:, 0]
                i += len(idx)
        return np.clip(out, -1.0, 1.0)

    # ---- contract -----------------------------------------------------------------------
    def fit(self, prompts, y, val_prompts=None, val_y=None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        t0 = time.perf_counter()
        c = self.config
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        self.device = _device()

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

        best_mse, best_state, bad = np.inf, None, 0
        for epoch in range(1, c["max_epochs"] + 1):
            te = time.perf_counter()
            self.model.train()
            loss_sum, padded = 0.0, 0
            for idx in epoch_batches(epoch):
                padded += len(idx) * max(lengths[i] for i in idx)
                opt.zero_grad(set_to_none=True)
                loss_sum += self._train_step([ids[i] for i in idx], [labels[i] for i in idx], len(idx))
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), c["max_grad_norm"])
                opt.step()
                sched.step()
            rec = dict(epoch=epoch, train_loss=loss_sum / len(ids), epoch_train_seconds=time.perf_counter() - te,
                       batches=steps_per_epoch, oom_splits=self.oom_splits,
                       padding_fraction=1.0 - sum(lengths) / max(1, padded))
            if val_ids is None:
                self.history.append(rec)
                continue
            mse = float(np.mean((self._gains(val_ids) - val_y) ** 2))
            rec["val_advantage_mse"] = mse
            self.history.append(rec)
            print(f"    epoch {epoch}: train loss {rec['train_loss']:.4f}  val advantage MSE {mse:.5f}  "
                  f"({rec['epoch_train_seconds']:.0f}s, {self.device.type})", flush=True)
            if mse < best_mse - 1e-12:
                best_mse, bad, self.best_epoch = mse, 0, epoch
                best_state = {k: v.detach().to("cpu", copy=True) for k, v in self.model.state_dict().items()}
            else:
                bad += 1
                if bad >= c["patience"]:
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        else:
            self.best_epoch = len(self.history)
        self.model.eval()
        self.fit_seconds = time.perf_counter() - t0
        return self

    def predict_gain(self, prompts) -> np.ndarray:
        return self._gains(self._encode(prompts)).astype(float)

    # ---- optional hooks -----------------------------------------------------------------
    def extra_info(self) -> dict:
        import torch
        dev = getattr(self, "device", None)
        gpu = torch.cuda.get_device_name(0) if dev is not None and dev.type == "cuda" else None
        return dict(history=self.history, best_epoch=self.best_epoch, device=dev.type if dev else None, gpu=gpu,
                    precision=self.precision if dev else None, truncation_rate_fit=self.truncation_rate_fit,
                    oom_splits=self.oom_splits, full_config=dict(self.config))

    def save(self, path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        (path / "router_info.json").write_text(json.dumps(self.extra_info(), indent=2))

    def release(self) -> None:
        dev = getattr(self, "device", None)
        self.model = None
        if dev is not None:
            _empty_cache(dev)


# The declared search: learning rate only (docs/PROTOCOL.md, section 5). Fixed before the first
# full run; do not change after results have been seen.
ENCODER_GRID = [dict(lr=2e-5), dict(lr=5e-5)]
