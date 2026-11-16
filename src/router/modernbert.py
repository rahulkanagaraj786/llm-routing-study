"""ModernBERT candidate: fine-tuned encoder + three-class head.

Labels: Y=-1 -> class 0, Y=0 -> class 1, Y=+1 -> class 2 (fixed order, saved with the model).
gain = P(class 2) - P(class 0). Inputs are prompts only.
Runs on CUDA, Apple MPS, or CPU; the device actually used is recorded.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

MODEL_NAME = "answerdotai/ModernBERT-base"
MODEL_REVISION = "8949b909ec900327062f0ebf497f51aef5e6f0c8"  # pinned Hugging Face commit
LABEL_MAP = {-1: 0, 0: 1, 1: 2}
SEED = 221

DEFAULTS = dict(lr=3e-5, max_epochs=3, batch_size=16, max_length=1024, weight_decay=0.01, warmup_frac=0.1,
                patience=1, grad_accum=1, token_budget=None)
# Declared search grid (learning rate only; epochs are chosen by early stopping).
MODERNBERT_GRID = [dict(lr=2e-5), dict(lr=5e-5)]
# Settings that fit this project's 16 GB Apple-silicon Mac under a 0.5 MPS memory watermark
# (measured in results/audit/modernbert_smoke_*.json): at most 8 prompts and at most
# 1536 padded tokens per batch, prompts truncated to 512 tokens. (2048 ran out of memory
# in the second epoch of the first real fit; see the deviation log in docs/PROTOCOL.md.)
LOCAL_MPS = dict(batch_size=8, max_length=512, token_budget=1536)
# Settings intended for a CUDA GPU with >= 16 GB memory.
REMOTE_GPU = dict(batch_size=16, max_length=1024, token_budget=None)


def pick_device():
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class ModernBertRouter:
    family = "modernbert"

    def __init__(self, **config):
        self.config = {**DEFAULTS, **config}
        self.history: list[dict] = []
        self.device = None

    # ---- data helpers -------------------------------------------------
    def _load(self):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        self.device = pick_device()
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, revision=MODEL_REVISION, num_labels=3)
        self.model.to(self.device)

    def _encode(self, prompts):
        enc = self.tokenizer(list(prompts), add_special_tokens=True, truncation=False)["input_ids"]
        L = self.config["max_length"]
        full = np.array([len(e) for e in enc])
        # keep [CLS] + first tokens + [SEP] when truncating (head truncation)
        ids = [e if len(e) <= L else e[: L - 1] + [e[-1]] for e in enc]
        return ids, float((full > L).mean())

    def _padded(self, n_tokens: int) -> int:
        return min(-(-n_tokens // 64) * 64, self.config["max_length"])  # round up to a multiple of 64

    def _batches(self, ids, batch_size, shuffle, rng=None, token_budget=None):
        """Length-bucketed batches. With a token budget, a batch also stops growing when
        (batch rows) x (padded length) would exceed the budget, so long prompts get smaller batches."""
        n = len(ids)
        if shuffle:
            order = rng.permutation(n)
            chunk = batch_size * 50  # sort by length inside chunks: less padding, still shuffled
            order = np.concatenate([sorted(order[i:i + chunk], key=lambda j: len(ids[j])) for i in range(0, n, chunk)])
        else:
            order = np.argsort([len(x) for x in ids], kind="stable")
        batches, cur, cur_max = [], [], 0
        for j in order:
            m = max(cur_max, len(ids[j]))
            too_big = token_budget is not None and (len(cur) + 1) * self._padded(m) > token_budget
            if cur and (len(cur) >= batch_size or too_big):
                batches.append(np.array(cur)); cur, m = [], len(ids[j])
            cur.append(int(j)); cur_max = m
        if cur:
            batches.append(np.array(cur))
        if shuffle:
            batches = [batches[i] for i in rng.permutation(len(batches))]
        return batches

    def _collate(self, ids, idx):
        import torch
        pad = self.tokenizer.pad_token_id
        m = self._padded(max(len(ids[j]) for j in idx))
        x = torch.full((len(idx), m), pad, dtype=torch.long)
        a = torch.zeros((len(idx), m), dtype=torch.long)
        for r, j in enumerate(idx):
            x[r, : len(ids[j])] = torch.tensor(ids[j])
            a[r, : len(ids[j])] = 1
        return x.to(self.device), a.to(self.device)

    def _autocast(self):
        import torch
        if self.device.type == "cuda":
            return torch.autocast("cuda", dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
        import contextlib
        return contextlib.nullcontext()

    # ---- training -----------------------------------------------------
    def fit(self, prompts, y, val_prompts, val_y, max_steps: int | None = None, log=print):
        """Train with cross-entropy; after each epoch score the validation prompts and keep the
        epoch with the lowest squared error between gain and Y (early stopping, patience from config)."""
        import torch
        t0 = time.perf_counter()
        self._load()
        c = self.config
        ids, self.train_truncation_rate = self._encode(prompts)
        labels = np.array([LABEL_MAP[int(v)] for v in y])
        val_y = np.asarray(val_y, dtype=float)
        rng = np.random.default_rng(SEED)
        steps_per_epoch = int(np.ceil(len(self._batches(ids, c["batch_size"], True, np.random.default_rng(0), c["token_budget"])) / c["grad_accum"]))
        total = steps_per_epoch * c["max_epochs"]
        no_decay = ("bias", "norm")
        params = [
            {"params": [p for n, p in self.model.named_parameters() if not any(k in n.lower() for k in no_decay)],
             "weight_decay": c["weight_decay"]},
            {"params": [p for n, p in self.model.named_parameters() if any(k in n.lower() for k in no_decay)],
             "weight_decay": 0.0},
        ]
        opt = torch.optim.AdamW(params, lr=c["lr"])
        warm = max(1, int(c["warmup_frac"] * total))
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (total - s) / max(1, total - warm)))
        scaler = torch.amp.GradScaler("cuda", enabled=self.device.type == "cuda")
        best, best_state, bad, step = None, None, 0, 0
        for epoch in range(1, c["max_epochs"] + 1):
            self.model.train()
            te, tl, nb = time.perf_counter(), 0.0, 0
            opt.zero_grad(set_to_none=True)
            for bi, idx in enumerate(self._batches(ids, c["batch_size"], True, rng, c["token_budget"])):
                last = (bi + 1) % c["grad_accum"] == 0
                tl += self._train_step(ids, labels, idx, opt, sched, scaler, last)
                nb += 1
                if self.device.type == "mps" and nb % 25 == 0:
                    torch.mps.empty_cache()
                if max_steps and nb >= max_steps:
                    break
            train_seconds = time.perf_counter() - te
            if self.device.type == "mps":
                torch.mps.empty_cache()
            gain = self.predict_gain(val_prompts)
            mse = float(np.mean((gain - val_y) ** 2))
            rec = dict(epoch=epoch, train_loss=tl / nb, val_advantage_mse=mse, epoch_train_seconds=train_seconds,
                       batches=nb, oom_splits=getattr(self, "oom_splits", 0))
            self.history.append(rec)
            log(f"    epoch {epoch}: train_loss {rec['train_loss']:.4f}  val_advantage_mse {mse:.5f}  {train_seconds:.0f}s")
            if best is None or mse < best - 1e-9:
                best, bad = mse, 0
                best_state = {k: v.detach().to("cpu", copy=True) for k, v in self.model.state_dict().items()}
                self.best_epoch = epoch
            else:
                bad += 1
                if bad > c["patience"] - 1:
                    break
            if max_steps:
                break
        self.model.load_state_dict(best_state)
        self.model.to(self.device)
        self.val_advantage_mse = best
        self.fit_seconds = time.perf_counter() - t0
        return self

    def _train_step(self, ids, labels, idx, opt, sched, scaler, do_update=True) -> float:
        """One forward/backward (and optimizer update). On an out-of-memory error the batch is
        split in half and each half is taken as its own step; the count is kept in self.oom_splits."""
        import torch
        c = self.config
        try:
            x, a = self._collate(ids, idx)
            t = torch.tensor(labels[idx], device=self.device)
            with self._autocast():
                loss = torch.nn.functional.cross_entropy(self.model(input_ids=x, attention_mask=a).logits.float(), t)
            scaler.scale(loss / c["grad_accum"]).backward()
            if do_update:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); sched.step()
                opt.zero_grad(set_to_none=True)
            return float(loss.detach())
        except RuntimeError as e:
            if "out of memory" not in str(e) or len(idx) == 1 or c["grad_accum"] != 1:
                raise
        # recover outside the except block so the failed graph can be freed
        x = a = t = loss = None
        opt.zero_grad(set_to_none=True)
        if self.device.type == "mps":
            torch.mps.empty_cache()
        elif self.device.type == "cuda":
            torch.cuda.empty_cache()
        self.oom_splits = getattr(self, "oom_splits", 0) + 1
        h = len(idx) // 2
        return 0.5 * (self._train_step(ids, labels, idx[:h], opt, sched, scaler) +
                      self._train_step(ids, labels, idx[h:], opt, sched, scaler))

    def predict_gain(self, prompts, batch_size: int = 32) -> np.ndarray:
        import torch
        ids, self.last_truncation_rate = self._encode(prompts)
        out = np.zeros(len(ids))
        self.model.eval()
        with torch.no_grad():
            budget = self.config["token_budget"]
            for idx in self._batches(ids, batch_size, False, token_budget=budget):
                x, a = self._collate(ids, idx)
                with self._autocast():
                    p = torch.softmax(self.model(input_ids=x, attention_mask=a).logits.float(), dim=-1)
                out[idx] = (p[:, 2] - p[:, 0]).cpu().numpy()
        if self.device.type == "mps":
            torch.mps.empty_cache()
        return out

    def save(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        (path / "router_meta.json").write_text(json.dumps(dict(
            base_model=MODEL_NAME, revision=MODEL_REVISION, label_map={str(k): v for k, v in LABEL_MAP.items()},
            gain="softmax(logits)[2] - softmax(logits)[0]", config=self.config, history=self.history,
            best_epoch=getattr(self, "best_epoch", None), device=str(self.device)), indent=2))
