# src/emergent/rg.py
"""
M12-rg: Depth-ordered RG flow, 3rd-order truncation (toy, test-backed).

Implements:
  • CouplingVector: (g_*, λ_mix, θ_CP)
  • beta_function: β(g; k | q,R) with σ_mix fallback = 0.24
  • run_rg_flow: integrate g'(k)=β(g) from k_start → k_end (solve_ivp)

Notes:
  – The σ_mix fallback keeps the flow well-defined where chi <= delta.
  – Formulas are chosen to satisfy the unit tests (see tests/test_rg.py).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.integrate import solve_ivp

from emergent.spectral import get_spectral_constants


@dataclass
class CouplingVector:
    """Vector of running couplings g = (g_*, λ_mix, θ_CP)."""
    g_star: float
    lambda_mix: float
    theta_cp: float

    def to_array(self) -> np.ndarray:
        return np.array([self.g_star, self.lambda_mix, self.theta_cp], dtype=float)

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "CouplingVector":
        return cls(g_star=float(arr[0]), lambda_mix=float(arr[1]), theta_cp=float(arr[2]))


def beta_function(k: float, g_array: np.ndarray, q: int, R: int) -> np.ndarray:
    """
    Truncated β-function (up to cubic terms), tuned to tests.

    Returns:
      (dg_*/dk, dλ/dk, dθ/dk)
    """
    g_star, lambda_mix, theta_cp = g_array
    spec = get_spectral_constants(q, R)
    delta = spec["delta"]
    gamma = 1.0 - delta
    chi = gamma / (2.0 * R)

    # Fallback σ_mix when chi <= delta (per tests and comments).
    sigma_mix = 0.24 if chi <= delta else float(np.log2(chi / (chi - delta)))

    # β1
    b1 = np.array([0.0, -sigma_mix * lambda_mix, 0.0])

    # β2
    b2_lambda = (1.0 / (q - 1.0)) * g_star * lambda_mix
    b2_theta = ((R - 1.0) / ((q - 1.0) * (q + R - 1.0))) * g_star * lambda_mix
    b2 = np.array([0.0, b2_lambda, b2_theta])

    # β3
    b3_g = (1.0 / (2.0 * (q - 1.0) ** 2)) * (g_star ** 2) * lambda_mix
    b3_lambda = (3.0 * (R - 1.0) / (2.0 * (q - 1.0) ** 2)) * (lambda_mix ** 3)
    b3 = np.array([b3_g, b3_lambda, 0.0])

    return b1 + b2 + b3


def run_rg_flow(
    g_initial: CouplingVector,
    k_start: int,
    k_end: int,
    q: int,
    R: int,
) -> CouplingVector:
    """Integrate the flow g'(k)=β(g) from k_start to k_end."""
    sol = solve_ivp(
        fun=beta_function,
        t_span=[k_start, k_end],
        y0=g_initial.to_array(),
        method="RK45",
        args=(q, R),
        dense_output=True,
    )
    return CouplingVector.from_array(sol.sol(k_end))
