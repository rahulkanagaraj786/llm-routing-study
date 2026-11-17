"""The contract every router model follows.

A router model turns prompt text into a `gain`: the expected accuracy advantage of GPT-5
over Flash on that prompt, a number between -1 and +1. Everything else in the project
(cutoff tuning, evaluation, reporting) is shared and only ever calls the methods below.

To add a model: copy `gbm.py`, change the class, and add one entry to `registry.py`.
"""
from __future__ import annotations

import numpy as np


class Router:
    #: short name used on the command line and for the results folder, e.g. "gbm"
    family: str = ""

    def __init__(self, **config):
        #: the settings of this particular fit; must be JSON-serialisable
        self.config: dict = dict(config)
        #: wall-clock seconds spent in fit(); set by fit()
        self.fit_seconds: float = 0.0

    def fit(self, prompts, y, val_prompts=None, val_y=None) -> "Router":
        """Learn from training prompts and their labels.

        prompts      array of str. The ONLY input a model may use to predict.
        y            array of -1 / 0 / +1  (GPT-5 correct minus Flash correct).
        val_prompts, val_y
                     validation prompts and labels. Models that train in epochs may use them
                     for early stopping; other models ignore them. Never fit preprocessing
                     (vocabulary, scaling, ...) on them.
        Must set self.fit_seconds and return self.
        """
        raise NotImplementedError

    def predict_gain(self, prompts) -> np.ndarray:
        """Return one gain per prompt, each in [-1, 1].

        For a three-class model: gain = P(Y = +1) - P(Y = -1).
        For a regression model: the predicted average of Y."""
        raise NotImplementedError

    # ---- optional hooks -------------------------------------------------------------
    def extra_info(self) -> dict:
        """Anything worth recording with the results (iterations, best epoch, device...)."""
        return {}

    def save(self, path) -> None:
        """Save the fitted model under `path` (a folder). Default: one joblib file."""
        import joblib
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path / "router.joblib")

    def release(self) -> None:
        """Free memory after use (GPU models override this)."""
