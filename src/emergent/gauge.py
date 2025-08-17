# src/emergent/gauge.py
"""
M10-gauge-chirality: Symbolic SM content and anomaly cancellations.

Provides:
  • get_sm_field_content: left-handed Weyl field list per generation.
  • check_all_anomalies: [U(1)_Y]^3, Grav^2–Y, SU(2)^2–Y, SU(3)^2–Y traces.

Notes
-----
• Purely symbolic, finite checks. The choices of reps/hypercharges match the
  usual SM assignments; overall normalization factors drop out when testing
  *cancellation* (0 value) of the anomaly traces.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class WeylField:
    """Left-handed Weyl field with its SM quantum numbers."""
    name: str
    su3_dim: int  # 1 or 3
    su2_dim: int  # 1 or 2
    hypercharge_Y: float  # U(1)_Y

    @property
    def multiplicity(self) -> int:
        """Total number of internal states for this field."""
        return int(self.su3_dim * self.su2_dim)


def get_sm_field_content(n_generations: int = 1) -> List[WeylField]:
    """Return SM content repeated n_generations times."""
    if n_generations < 1:
        raise ValueError("Number of generations must be at least 1.")
    one_gen = [
        # Quarks
        WeylField("Q_L",    3, 2,  1/6),
        WeylField("u_R^c",  3, 1, -2/3),
        WeylField("d_R^c",  3, 1,  1/3),
        # Leptons
        WeylField("L_L",    1, 2, -1/2),
        WeylField("e_R^c",  1, 1,  1.0),
        # Optional sterile (anomaly-free):
        # WeylField("nu_R^c", 1, 1, 0.0),
    ]
    return one_gen * n_generations


def check_all_anomalies(n_generations: int = 1) -> Dict[str, float]:
    """Compute anomaly traces; zero indicates cancellation."""
    fields = get_sm_field_content(n_generations)

    # [U(1)_Y]^3 anomaly: sum(multiplicity * Y^3)
    a_y3  = sum(f.multiplicity * (f.hypercharge_Y ** 3) for f in fields)

    # Gravitational^2–U(1)_Y anomaly: sum(multiplicity * Y)
    a_ggY = sum(f.multiplicity *  f.hypercharge_Y        for f in fields)

    # SU(2)^2–U(1)_Y: sum over SU(2) doublets of (d_c * Y)
    a_su2_Y = sum(f.su3_dim * f.hypercharge_Y for f in fields if f.su2_dim == 2)

    # SU(3)^2–U(1)_Y: sum over SU(3) triplets of (d_w * Y)
    a_su3_Y = sum(f.su2_dim * f.hypercharge_Y for f in fields if f.su3_dim == 3)

    return {
        "Y^3": a_y3,
        "Grav^2 * Y": a_ggY,
        "SU(2)^2 * Y": a_su2_Y,
        "SU(3)^2 * Y": a_su3_Y,
    }
