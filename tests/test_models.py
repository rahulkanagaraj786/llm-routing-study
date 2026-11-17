"""Every written model must follow the contract in src/router/base.py. Synthetic text only."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from router import registry
from router.base import Router


def synthetic(n=240, seed=0):
    rng = np.random.default_rng(seed)
    words = ["prove", "theorem", "capital", "river", "function", "return", "integral", "who", "year", "city"]
    prompts = np.array([" ".join(rng.choice(words, size=int(rng.integers(4, 12)))) + "?" for _ in range(n)], dtype=object)
    y = np.array([1 if "prove" in p else (-1 if "river" in p and rng.random() < 0.3 else 0) for p in prompts])
    return prompts, y


def test_written_models_follow_the_contract():
    prompts, y = synthetic()
    tr, va = slice(0, 180), slice(180, 240)
    for family in registry.names():
        if registry.FAMILIES[family]["cls"] is None:
            continue  # planned, not written yet
        grid = registry.grid(family)
        assert grid and all(isinstance(c, dict) for c in grid), family
        r = registry.make(family, grid[0])
        assert isinstance(r, Router) and r.family == family
        assert r.fit(prompts[tr], y[tr], prompts[va], y[va]) is r and r.fit_seconds > 0
        g = r.predict_gain(prompts[va])
        assert g.shape == (60,) and np.isfinite(g).all() and (np.abs(g) <= 1 + 1e-6).all(), family
        import json; json.dumps(r.config); json.dumps(r.extra_info())
        r.release()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
