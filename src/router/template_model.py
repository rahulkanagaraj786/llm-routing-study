"""Skeleton for a new router model. Copy this file, rename it, and fill in the two methods.

Worked example to read first: LogisticRouter in models.py.
Full instructions: docs/TEAM_GUIDE.md.

Steps:
  1. Copy this file, e.g. to src/router/gbm.py, and rename the class.
  2. Fill in fit() and predict_gain().
  3. List the settings you want to try in MY_GRID. Keep it small, and fix it before your
     first real run: it must not change after you have seen results.
  4. Point your entry in src/router/registry.py at the class and the grid.
  5. python scripts/05_dev_compare.py <your model name> --smoke
"""
from __future__ import annotations

import time

import numpy as np

from .base import Router


class MyRouter(Router):
    family = "change-me"   # must match your key in registry.py

    def __init__(self, some_setting: float = 1.0):
        # pass every setting to super().__init__ so it is recorded with the results
        super().__init__(some_setting=some_setting)

    def fit(self, prompts, y, val_prompts=None, val_y=None):
        """prompts: array of question texts.  y: array of -1 / 0 / +1.

        Rules:
          - Use only `prompts` as input. No scores, costs, answers or benchmark names.
          - Fit all preprocessing (vocabulary, scaling, ...) on these training prompts only.
          - val_prompts / val_y may be used for early stopping only (epoch-based models).
        """
        t0 = time.perf_counter()
        # 1. turn prompts into features
        #    (TextFeatures in models.py gives the same word-count features logistic regression uses)
        # 2. fit your model on (features, y)
        raise NotImplementedError("write fit()")
        self.fit_seconds = time.perf_counter() - t0
        return self

    def predict_gain(self, prompts) -> np.ndarray:
        """Return one number per prompt, between -1 and +1:
        gain = P(Y = +1) - P(Y = -1)     for a three-class model, or
        gain = predicted average of Y    for a regression model."""
        raise NotImplementedError("write predict_gain()")


# The settings to try. One dict per setting; keys must match __init__ arguments.
MY_GRID = [dict(some_setting=v) for v in (0.5, 1.0, 2.0)]
