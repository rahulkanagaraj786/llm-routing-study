"""The list of router models in the study, and who owns each.

Every script (development comparison, selection, final evaluation, reports) reads this list.
A model that is planned but not written yet has cls=None; its owner fills in the two paths.
"""
from __future__ import annotations

import importlib

# order: from simplest to most complex. The selection rule prefers the simpler router unless
# a more complex one is clearly better (docs/PROTOCOL.md, section 6).
#
# tune_every_rotation: True  = try every setting in each of the five rotations (cheap models)
#                      False = choose the setting in the first rotation and reuse it (slow models)
FAMILIES = {
    "logistic": dict(
        order=0, label="Regularized logistic regression", owner="Rahul",
        cls="router.models:LogisticRouter", grid="router.models:LOGISTIC_GRID", tune_every_rotation=True),
    "gbm": dict(
        order=1, label="Gradient boosting", owner="to be assigned",
        cls=None, grid=None, tune_every_rotation=True),            # e.g. "router.gbm:GradientBoostingRouter", "router.gbm:GBM_GRID"
    "encoder": dict(
        order=2, label="Fine-tuned text encoder (ModernBERT-base)", owner="Priyanka",
        cls="router.encoder:EncoderRouter", grid="router.encoder:ENCODER_GRID", tune_every_rotation=False),
}


def _load(spec: str):
    module, name = spec.split(":")
    return getattr(importlib.import_module(module), name)


def names() -> list[str]:
    """All families, simplest first."""
    return sorted(FAMILIES, key=lambda f: FAMILIES[f]["order"])


def check(family: str) -> None:
    if family not in FAMILIES:
        raise SystemExit(f"unknown model '{family}'. Known models: {', '.join(names())}")
    if FAMILIES[family]["cls"] is None:
        raise SystemExit(f"'{family}' has not been written yet. Its owner copies src/router/template_model.py, "
                         "fills it in, and sets cls and grid for it in src/router/registry.py (see docs/TEAM_GUIDE.md).")


def make(family: str, config: dict):
    """Build one router of this family with the given settings."""
    check(family)
    return _load(FAMILIES[family]["cls"])(**config)


def grid(family: str) -> list[dict]:
    """The declared settings to try for this family."""
    check(family)
    return list(_load(FAMILIES[family]["grid"]))


def label(family: str) -> str:
    return FAMILIES[family]["label"]
