# src/emergent/physics_maps.py
"""
Physics map hooks (Phase E).

This module defines a typed, pluggable interface ("Hooks") for supplying
paper-derived maps to the prediction pipeline. By default we provide
conservative, numerically well-behaved fallbacks. You can swap them by
calling make_hooks_from_module("emergent.paper_maps.v8") in notebooks/CLI.

Conventions:
- g1 is hypercharge (U(1)_Y) *in SM-normalization* (no 5/3 factor baked-in).
- g2 is SU(2)_L.
- sin^2 θ_W = g1^2 / (g1^2 + g2^2).
- α_EM   = e^2 / (4π), with e^2 = g1^2 * g2^2 / (g1^2 + g2^2).
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Callable, Optional, Tuple


# -------------------------- type aliases --------------------------

# (g_star, lambda_mix, q, R, k) -> (g1, g2)
GaugeMap = Callable[[float, float, int, int, float], Tuple[float, float]]
# dS/dt -> Lambda
CosmoMap = Callable[[float], float]
# (g_star, lambda_mix, theta_cp) -> EDM value [e·cm]
EDMMap = Callable[[float, float, float], float]
# Direct (q,R) -> Lambda
CosmoMapFromQR = Callable[[int, int], float]


@dataclass(frozen=True)
class Hooks:
    gauge_couplings: GaugeMap
    lambda_from_entropy: CosmoMap
    edm_from_rg: EDMMap
    # Optional: preferred direct cosmology hook from (q,R)
    lambda_from_qR: Optional[CosmoMapFromQR] = None


# ---------------------- conservative defaults ---------------------

def _default_gauge_couplings(g_star: float, lambda_mix: float, q: int, R: int, k: float) -> Tuple[float, float]:
    """
    Default k-dependent map with mild, monotone running sensitivity.
    Replace in emergent.paper_maps.v8 with the paper’s final map.

    We set g2 ≈ g_star, and drive g1 by a smooth function of (g_star, lambda_mix, k).
    The form is chosen so sin^2θ_W sits in a realistic (0.2–0.5) range without
    ad-hoc normalizers.
    """
    g2 = max(1e-6, float(g_star))
    # A gentle coupling “mix”: at high k, g1≈g2; at low k, g1 slightly smaller.
    # The factors are tame to preserve numerical stability in tests.
    roll = 1.0 / (1.0 + 0.02 * k)         # decreases with k
    g1 = max(1e-6, float(0.8 * g2 + 0.25 * lambda_mix * roll))
    return g1, g2


def _default_lambda_from_entropy(dS_dt: float) -> float:
    # Simple proportional placeholder; can be ignored if lambda_from_qR is provided.
    return float(max(0.0, dS_dt))


def _default_edm_from_rg(g_star: float, lambda_mix: float, theta_cp: float) -> float:
    # Conservative scaling used in tests; replace in paper_maps when available.
    return float(abs(theta_cp) * 1.0e-16)


default_hooks = Hooks(
    gauge_couplings=_default_gauge_couplings,
    lambda_from_entropy=_default_lambda_from_entropy,
    edm_from_rg=_default_edm_from_rg,
    lambda_from_qR=None,
)


def make_hooks_from_module(module_path: str) -> Hooks:
    """
    Dynamically import a hooks provider. The module must expose a callable
    `make_hooks() -> Hooks`.
    """
    mod = importlib.import_module(module_path)
    hooks = getattr(mod, "make_hooks")()
    if not isinstance(hooks, Hooks):
        raise TypeError(f"{module_path}.make_hooks() did not return a Hooks instance.")
    return hooks
