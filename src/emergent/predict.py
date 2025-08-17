# src/emergent/predict.py
"""
Prediction utilities and "cards" for Phase E.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from emergent.physics_maps import Hooks, default_hooks
from emergent.rg import CouplingVector, beta_function, run_rg_flow

# ----------------------------
# Data containers
# ----------------------------

@dataclass
class WeakMixCurve:
    k: np.ndarray
    mean: np.ndarray
    lo: np.ndarray
    hi: np.ndarray

@dataclass
class PredictionCard:
    title: str
    central: Dict[str, float]
    interval: Dict[str, Tuple[float, float]]

    def to_dict(self) -> Dict[str, object]:
        d: Dict[str, object] = {"title": self.title, "central": self.central}
        if self.interval:
            d["interval"] = self.interval
        return d

# ----------------------------
# Helper functions
# ----------------------------

def _sin2_thetaW_and_alpha(g1: float, g2: float) -> Tuple[float, float]:
    g1sq, g2sq = g1 * g1, g2 * g2
    denom = g1sq + g2sq
    if denom <= 0.0:
        return 0.0, 0.0
    sin2 = g1sq / denom
    e2 = (g1sq * g2sq) / denom
    alpha = e2 / (4.0 * np.pi)
    return float(sin2), float(alpha)

def _get_entropy_predictor():
    """Optional import for the fallback cosmology card."""
    try:
        # CORRECTED: Import from entropy_fp, not entropy_mc
        from emergent.entropy_fp import predict_entropy_trajectory
        return predict_entropy_trajectory
    except ImportError:
        return None

# ----------------------------
# Main prediction functions
# ----------------------------

def predict_weak_mixing_curve(
    g0: CouplingVector,
    *,
    q: int,
    R: int,
    k_start: float,
    k_end: float,
    n_grid: int = 101,
    bootstrap: int = 0,
    seed: int = 0,
    hooks: Optional[Hooks] = None,
    param_uncertainties: Optional[Dict[str, float]] = None,
) -> Tuple[WeakMixCurve, Dict[str, float]]:
    hooks = hooks or default_hooks
    rng = np.random.default_rng(seed)
    ks = np.linspace(float(k_start), float(k_end), int(n_grid))

    def single_run(g_init: CouplingVector) -> Tuple[np.ndarray, np.ndarray, CouplingVector]:
        gf = run_rg_flow(g_init, k_start, k_end, q, R)
        g_star_line = np.linspace(g_init.g_star, gf.g_star, n_grid)
        lmix_line = np.linspace(g_init.lambda_mix, gf.lambda_mix, n_grid)
        sin2 = np.empty(n_grid, dtype=float)
        for i, k in enumerate(ks):
            g1, g2 = hooks.gauge_couplings(g_star_line[i], lmix_line[i], q, R, float(k))
            s2_val, _ = _sin2_thetaW_and_alpha(g1, g2)
            sin2[i] = s2_val
        return ks, sin2, gf

    ks_c, sin2_c, g_final_c = single_run(g0)
    all_sin2 = [sin2_c]
    if bootstrap > 0:
        for _ in range(bootstrap):
            g_star_s = rng.normal(g0.g_star, param_uncertainties.get("g_star", 1e-6) if param_uncertainties else 1e-6)
            lmix_s = rng.normal(g0.lambda_mix, param_uncertainties.get("lambda_mix", 1e-6) if param_uncertainties else 1e-6)
            tcp_s = rng.normal(g0.theta_cp, param_uncertainties.get("theta_cp", 1e-6) if param_uncertainties else 1e-6)
            _, s2_b, _ = single_run(CouplingVector(g_star=g_star_s, lambda_mix=lmix_s, theta_cp=tcp_s))
            all_sin2.append(s2_b)
    
    arr = np.vstack(all_sin2)
    mean = np.mean(arr, axis=0)
    lo = np.quantile(arr, 0.16, axis=0) if bootstrap > 0 else mean
    hi = np.quantile(arr, 0.84, axis=0) if bootstrap > 0 else mean

    g1_EW, g2_EW = hooks.gauge_couplings(g_final_c.g_star, g_final_c.lambda_mix, q, R, float(ks_c[-1]))
    sin2_EW, alpha_EW = _sin2_thetaW_and_alpha(g1_EW, g2_EW)

    summary = {
        "sin2_thetaW_EW": sin2_EW, "alpha_EM_EW": alpha_EW, "theta_cp_EW": g_final_c.theta_cp,
        "g_star_EW": g_final_c.g_star, "lambda_mix_EW": g_final_c.lambda_mix,
    }
    return WeakMixCurve(ks_c, mean, lo, hi), summary


def make_card_weakmix(
    g0: CouplingVector,
    *,
    q: int, R: int, k_start: float, k_end: float,
    n_grid: int = 101, bootstrap: int = 16, hooks: Optional[Hooks] = None,
) -> "PredictionCard":
    curve, summary = predict_weak_mixing_curve(
        g0, q=q, R=R, k_start=k_start, k_end=k_end, n_grid=n_grid, bootstrap=bootstrap, hooks=hooks
    )
    return PredictionCard(
        title="Weak mixing prediction", central=summary,
        interval={"sin2_thetaW_band@EW": (float(curve.lo[-1]), float(curve.hi[-1]))},
    )


def make_card_cosmology(
    *,
    site_builder: Optional[Tuple[int, int, int, float, int]] = None,
    q: int = 6, R: int = 4, observer_depth: int = 3, ticks: int = 8,
    hooks: Optional[Hooks] = None,
) -> "PredictionCard":
    hooks = hooks or default_hooks
    dSdt = 0.0

    if hooks.lambda_from_qR is not None:
        lam = hooks.lambda_from_qR(q, R)
    else:
        predictor = _get_entropy_predictor()
        if predictor is None:
            raise RuntimeError("emergent.entropy_fp.predict_entropy_trajectory is required for fallback.")
        if site_builder is None:
            site_builder = (6, 6, 4, 0.6, 101)
        from emergent.poset import CausalSite
        n_layers, nodes_per_layer, R_loc, edge_prob, seed = site_builder
        rng = np.random.default_rng(seed)
        site = CausalSite.generate(n_layers=n_layers, nodes_per_layer=nodes_per_layer, R=R_loc, edge_prob=edge_prob, rng=rng)
        _, dS_pred = predictor(site, q, observer_depth, ticks)
        dSdt = float(np.nanmean(dS_pred)) if dS_pred else 0.0
        lam = hooks.lambda_from_entropy(dSdt)

    return PredictionCard(
        title="Cosmology (Λ proxy)", central={"Lambda_proxy": float(lam), "dS_dt": dSdt}, interval={}
    )


def make_card_edm(
    g0: CouplingVector,
    *,
    q: int, R: int, k_start: float, k_end: float,
    hooks: Optional[Hooks] = None,
) -> "PredictionCard":
    hooks = hooks or default_hooks
    gE = run_rg_flow(g0, k_start, k_end, q, R)
    dn = hooks.edm_from_rg(gE.g_star, gE.lambda_mix, gE.theta_cp)
    central = {"d_n_EDM_proxy": float(dn), "edm_n": float(dn), "theta_cp_EW": float(gE.theta_cp)}
    return PredictionCard(title="Neutron EDM proxy", central=central, interval={})


# -----------------------
# Export Helpers (FIX)
# -----------------------

def export_card_json(card: PredictionCard, path: str) -> None:
    """Saves a prediction card to a JSON file."""
    if (p_dir := os.path.dirname(path)) and p_dir:
        os.makedirs(p_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(card.to_dict(), f, indent=2)

def export_card_png(curve: WeakMixCurve, path: str) -> None:
    """Saves a prediction curve to a PNG file."""
    if (p_dir := os.path.dirname(path)) and p_dir:
        os.makedirs(p_dir, exist_ok=True)
    plt.figure(figsize=(6.2, 4.2))
    plt.fill_between(curve.k, curve.lo, curve.hi, alpha=0.25, label="±1σ")
    plt.plot(curve.k, curve.mean, label="mean")
    plt.xlabel("k (Energy Scale)")
    plt.ylabel("sin²(θ_W)")
    plt.title("Weak Mixing Prediction (RG Flow)")
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()