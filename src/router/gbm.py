"""Gradient boosting router for the study.

Same TextFeatures as logistic regression. Three-class classifier; gain = P(+1) - P(-1).
"""
from __future__ import annotations

import time

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

from .base import Router
from .models import SEED, TextFeatures


class GradientBoostingRouter(Router):
    """Multiclass gradient boosting; gain = P(Y=+1) - P(Y=-1)."""

    family = "gbm"

    def __init__(
        self,
        max_depth: int = 3,
        learning_rate: float = 0.1,
        use_stats: bool = False,
        n_estimators: int = 200,
        min_samples_leaf: int = 20,
        subsample: float = 0.8,
        max_features: str = "sqrt",
    ):
        super().__init__(
            max_depth=max_depth,
            learning_rate=learning_rate,
            use_stats=use_stats,
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            subsample=subsample,
            max_features=max_features,
        )

    def extra_info(self) -> dict:
        return dict(
            n_estimators=int(self.model.n_estimators_),
            n_terms=self.features.n_terms,
        )

    def fit(self, prompts, y, val_prompts=None, val_y=None):
        t0 = time.perf_counter()
        expected = {-1, 0, 1}
        actual = set(np.unique(y).tolist())
        if actual != expected:
            raise ValueError(f"Expected training classes {expected}, got {actual}")

        self.features = TextFeatures(use_stats=self.config["use_stats"]).fit(prompts)
        X = self.features.transform(prompts)
        c = self.config
        self.model = GradientBoostingClassifier(
            loss="log_loss",
            n_estimators=c["n_estimators"],
            learning_rate=c["learning_rate"],
            max_depth=c["max_depth"],
            min_samples_leaf=c["min_samples_leaf"],
            subsample=c["subsample"],
            max_features=c["max_features"],
            random_state=SEED,
        )
        self.model.fit(X, y)
        self.fit_seconds = time.perf_counter() - t0
        return self

    def predict_gain(self, prompts) -> np.ndarray:
        p = self.model.predict_proba(self.features.transform(prompts))
        cls = list(self.model.classes_)
        pos = p[:, cls.index(1)] if 1 in cls else 0.0
        neg = p[:, cls.index(-1)] if -1 in cls else 0.0
        gain = np.asarray(pos - neg, dtype=float)
        return np.clip(gain, -1.0, 1.0)


# Declared before the first full development run. Do not change after seeing results.
GBM_GRID = [
    dict(max_depth=d, learning_rate=lr, use_stats=s)
    for s in (False, True)
    for d in (3, 5)
    for lr in (0.05, 0.1)
]
