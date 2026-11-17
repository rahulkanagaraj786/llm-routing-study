"""Shared text features, and the reference model: regularized multinomial logistic regression.

Both take prompt text only. Every learned preprocessing step (vocabulary, IDF,
scaling) is fit on the training prompts passed to `fit` and nothing else.
"""
from __future__ import annotations

import re
import time

import numpy as np
import scipy.sparse as sp
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression

from .base import Router

SEED = 221
_CODE = re.compile(r"```|\bdef \w+\(|\bclass \w+|\bimport \w+|#include|\breturn\b|;\s*\n")
_CHOICES = re.compile(r"(?m)^\s*\(?[A-J][\.\)]\s")
_LATEX = re.compile(r"\$|\\frac|\\sum|\\int|\\left|\\begin\{")
_MATH_CHARS = set("+-*/=^<>%")
STAT_NAMES = ["log_chars", "log_words", "log_lines", "digit_frac", "math_symbol_frac", "upper_frac",
              "has_code", "has_choices", "has_latex", "n_question_marks"]


def simple_stats(prompts) -> np.ndarray:
    """Hand-built prompt statistics from the proposal (length, digits, math/code indicators)."""
    out = np.zeros((len(prompts), len(STAT_NAMES)))
    for i, p in enumerate(prompts):
        n = max(len(p), 1)
        out[i] = [np.log1p(len(p)), np.log1p(len(p.split())), np.log1p(p.count("\n") + 1),
                  sum(ch.isdigit() for ch in p) / n, sum(ch in _MATH_CHARS for ch in p) / n,
                  sum(ch.isupper() for ch in p) / n, float(bool(_CODE.search(p))), float(bool(_CHOICES.search(p))),
                  float(bool(_LATEX.search(p))), min(p.count("?"), 5)]
    return out


class TextFeatures:
    """Word 1-2 gram TF-IDF (sublinear tf, min_df=3, top `max_features` terms) plus optional
    standardized simple statistics.

    The pilot used TfidfVectorizer(max_features=12000), which breaks frequency ties at the
    cut-off with an unstable sort, so the vocabulary differs between machines. Here ties are
    broken by the term itself, which makes the vocabulary deterministic."""

    def __init__(self, max_features: int = 12000, min_df: int = 3, use_stats: bool = False):
        self.max_features, self.min_df, self.use_stats = max_features, min_df, use_stats

    def fit(self, prompts):
        cv = CountVectorizer(min_df=self.min_df, ngram_range=(1, 2))
        counts = cv.fit_transform(prompts)
        terms = np.array(cv.get_feature_names_out())
        tf = np.asarray(counts.sum(axis=0)).ravel()
        order = sorted(range(len(terms)), key=lambda j: (-int(tf[j]), terms[j]))[: self.max_features]
        vocab = sorted(terms[j] for j in order)
        self._cv = CountVectorizer(ngram_range=(1, 2), vocabulary=vocab)
        self._tfidf = TfidfTransformer(sublinear_tf=True).fit(self._cv.transform(prompts))
        self.n_terms = len(vocab)
        if self.use_stats:
            s = simple_stats(prompts)
            self._mu, self._sd = s.mean(axis=0), s.std(axis=0)
            self._sd[self._sd == 0] = 1.0
        return self

    def transform(self, prompts):
        X = self._tfidf.transform(self._cv.transform(prompts))
        if self.use_stats:
            s = (simple_stats(prompts) - self._mu) / self._sd
            X = sp.hstack([X, sp.csr_matrix(s)], format="csr")
        return X


class LogisticRouter(Router):
    """Three-class multinomial logistic regression; gain = P(Y=+1) - P(Y=-1).
    C is the inverse strength of the ridge (L2) penalty: smaller C = stronger penalty."""
    family = "logistic"

    def __init__(self, C: float = 1.0, use_stats: bool = False):
        super().__init__(C=C, use_stats=use_stats)

    def extra_info(self) -> dict:
        return dict(n_iter=self.n_iter, n_terms=self.features.n_terms)

    def fit(self, prompts, y, val_prompts=None, val_y=None):
        t0 = time.perf_counter()
        self.features = TextFeatures(use_stats=self.config["use_stats"]).fit(prompts)
        self.model = LogisticRegression(C=self.config["C"], max_iter=5000).fit(self.features.transform(prompts), y)
        self.fit_seconds = time.perf_counter() - t0
        self.n_iter = int(self.model.n_iter_.max())
        return self

    def predict_gain(self, prompts) -> np.ndarray:
        p = self.model.predict_proba(self.features.transform(prompts))
        cls = list(self.model.classes_)
        pos = p[:, cls.index(1)] if 1 in cls else 0.0
        neg = p[:, cls.index(-1)] if -1 in cls else 0.0
        return np.asarray(pos - neg, dtype=float)


class ForestRouter(Router):
    """Random forest regressor on Y in {-1, 0, +1}: estimates E[Y | prompt] directly
    (squared-error trees, as in the pilot), not class probabilities.

    No longer a candidate (it is not in registry.py). Kept so the archived forest results
    can be reproduced, and as a second example of a model that follows the contract."""
    family = "forest"

    def __init__(self, n_estimators: int = 300, max_depth=None, min_samples_leaf: int = 10,
                 max_features: float = 0.1, use_stats: bool = False, n_jobs: int = 8):
        super().__init__(n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                         max_features=max_features, use_stats=use_stats)
        self.n_jobs = n_jobs

    def fit(self, prompts, y, val_prompts=None, val_y=None):
        t0 = time.perf_counter()
        self.features = TextFeatures(use_stats=self.config["use_stats"]).fit(prompts)
        c = self.config
        self.model = RandomForestRegressor(n_estimators=c["n_estimators"], max_depth=c["max_depth"],
                                           min_samples_leaf=c["min_samples_leaf"], max_features=c["max_features"],
                                           n_jobs=self.n_jobs, random_state=SEED)
        self.model.fit(self.features.transform(prompts), y)
        self.fit_seconds = time.perf_counter() - t0
        return self

    def predict_gain(self, prompts) -> np.ndarray:
        return np.asarray(self.model.predict(self.features.transform(prompts)), dtype=float)


# Declared search grids (fixed before any new fit; see docs/PROTOCOL.md).
LOGISTIC_GRID = [dict(C=C, use_stats=s) for s in (False, True) for C in (0.1, 0.3, 1.0, 3.0, 10.0)]
FOREST_GRID = (
    [dict(n_estimators=160, max_depth=20, min_samples_leaf=10, max_features=0.1, use_stats=False)]  # pilot setting
    + [dict(n_estimators=300, max_depth=None, min_samples_leaf=leaf, max_features=mf, use_stats=True)
       for leaf in (5, 20) for mf in (0.1, 0.3)]
)
