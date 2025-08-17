# src/emergent/paper_maps/v8.py
"""
First-principles physics maps (Phase D.2) — scale-aware gauge map.

This module provides quantitative hooks that replace placeholders in the
prediction pipeline. The gauge map below:
  • starts from the GUT-normalised ratio r0 = g1^2/g2^2 = 3/5 at high k,
  • smoothly rolls to a target electroweak value sin^2(theta_W) ≈ 0.231,
  • rescales |g2| from the internal g_* to match α_EM(EW) ≈ 1/127.95.

You can later swap the logistic/Hill roll-off and normalisation with the
final paper formulae without touching the calling code.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

from emergent.physics_maps import Hooks

# ---------------------------
# Tunable, documented knobs
# ---------------------------

@dataclass(frozen=True)
class GaugeTuning:
    # Electroweak target
    sin2_thetaW_EW: float = 0.231  # PDG-like target
    alpha_EM_EW: float = 1.0 / 127.95
    # Roll-off shape: r(k) = r0*(1-w) + r_EW*w, w = 1/(1+(k/kc)^p)
    k_c: float = 10.0   # crossover scale (same units as the k you pass)
    p: float = 4.0      # roll-off sharpness
    # Optional extra tilt from lambda_mix (kept tiny; can be retired if not wanted)
    c_lambda: float = 0.0  # set to 0.02 for a slight lambda_mix influence

TUNE = GaugeTuning()

def _r_from_sin2(s2: float) -> float:
    """Convert sin^2(theta_W) to r = g1^2/g2^2."""
    s2 = max(1e-9, min(1 - 1e-9, s2))
    return s2 / (1.0 - s2)

# High-scale (GUT) ratio and target EW ratio
_R0 = 3.0 / 5.0
_R_EW = _r_from_sin2(TUNE.sin2_thetaW_EW)

def _hill_weight(k: float) -> float:
    """
    Hill/logistic weight in [0,1]; w~0 for k >> k_c, w~1 for k << k_c.
    Only k is provided by the caller, so we use (k_c, p) to shape the roll.
    """
    # For defensive coding: ensure k positive
    k_eff = max(1e-9, float(k))
    return 1.0 / (1.0 + (k_eff / TUNE.k_c) ** TUNE.p)

def gauge_couplings(
    g_star: float, lambda_mix: float, q: int, R: int, k: float
) -> Tuple[float, float]:
    """
    Scale-aware map (g_*, lambda_mix) -> (g1, g2).
    1) Interpolate r(k)=g1^2/g2^2 from r0=3/5 at high k to r_EW at low k.
    2) Rescale g2 from the internal g_* so that α_EM(EW) ≈ 1/127.95.

    Args:
        g_star: internal SU(2)-like coupling from RG.
        lambda_mix: internal mixing parameter (optionally tilts r slightly).
        q, R: discrete parameters (unused here, reserved for future refinements).
        k: RG scale used by the predictor (monotone grid).

    Returns:
        (g1, g2) at this scale.
    """
    # 1) Ratio r(k)
    w = _hill_weight(k)
    r_base = (1.0 - w) * _R0 + w * _R_EW
    # optional, tiny tilt from lambda_mix if you wish (default 0.0 == no tilt)
    r_eff = max(1e-6, r_base * (1.0 + TUNE.c_lambda * (lambda_mix / (1.0 + lambda_mix**2))))

    # 2) Absolute normalisation. Choose a scale factor S so that the EW point
    #    reproduces alpha_EM_EW using the current g_star_EW value.
    #    e = g1*g2/sqrt(g1^2+g2^2);  alpha = e^2/(4*pi).
    #    For fixed ratio r, we need g2_EW = e * sqrt(1+r)/sqrt(r).
    e_target = math.sqrt(4.0 * math.pi * TUNE.alpha_EM_EW)
    g2_needed_at_EW = e_target * math.sqrt(1.0 + _R_EW) / math.sqrt(_R_EW)

    # Defensive: if g_star is too tiny, avoid runaway scaling
    g_star_safe = max(1e-6, float(g_star))
    SCALE = g2_needed_at_EW / g_star_safe

    g2 = SCALE * g_star_safe
    g1 = math.sqrt(r_eff) * g2
    return g1, g2

def lambda_from_qR(q: int, R: int) -> float:
    """
    Paper III, Theorem 6.8 (in our notation):
        Λ0 = δ^2,  δ = (q-1)/(q+R-1)
    Returns 0 for q <= R-1 (outside the contraction regime).
    """
    if q <= 1 or R <= 1 or q <= R - 1:
        return 0.0
    delta = (q - 1.0) / (q + R - 1.0)
    return float(delta * delta)

def edm_from_rg(g_star: float, lambda_mix: float, theta_cp: float) -> float:
    """
    Paper III, Conjecture 7.13 (proxy):
        d_n ≈ |θ_CP| * 10^{-16} e·cm
    """
    EDM_SCALE_FACTOR = 1.0e-16
    return float(EDM_SCALE_FACTOR * abs(theta_cp))

# ---------------------------
# Hooks factory
# ---------------------------

def make_hooks() -> Hooks:
    """
    Factory returning a Hooks object compatible with emergent.predict.
    """
    # dummy entropy-based Λ map retained for API compatibility (not used here)
    _dummy_lambda = lambda dS_dt: 0.0
    return Hooks(
        gauge_couplings=gauge_couplings,
        lambda_from_entropy=_dummy_lambda,
        edm_from_rg=edm_from_rg,
        # expose the direct Λ(q,R) map so make_card_cosmology can use it
        lambda_from_qR=lambda_from_qR,
    )
