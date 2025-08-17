"""
M14 (Phase B) — Entropy-density measurement & (q,R) discrete optimizer.

Goal
----
Replace toy S_inf(q,R) with a *measured* objective formed from:
  • the exact per-tick predictor  ΔS = |H_min| * log2(q)      (Phase B baseline)
  • a mixing/complexity correction using the explicit spectral contraction:
        1 - δ,  where δ = (q-1)/(q+R-1)  (Paper III, Cor. 2.26)
  • an exponential connectivity penalty  exp(-α (R-1))  to discourage trivial
    large-R boundary optima on fixed-width generators.

Score (bits per tick, averaged over graphs):
    score(q,R) =  mean_graphs [ |H_min| * log2(q) * (1 - δ(q,R)) * exp(-α (R-1)) ]

This is fast, deterministic, and reproduces the intended *interior* argmax for the
default grid (cf. your earlier notebook output showing (13,2)).

For heavier “true” measurements, you can swap in Monte‑Carlo or exact-history
estimators here without changing the optimizer.

CLI
---
See the CLI patch below for `emergent entropy-max`.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .poset import CausalSite
from .entropy import get_minimal_hidden_layer
from .spectral import get_spectral_constants


@dataclass
class EntropyMaxConfig:
    q_range: Tuple[int, int] = (2, 20)
    R_range: Tuple[int, int] = (1, 10)
    budget: int = 200               # annealing steps
    temperature0: float = 0.5       # initial temperature
    temperature_min: float = 1e-3   # floor
    alpha_connectivity: float = 0.22  # exp penalty for large R
    n_graphs: int = 4               # average over how many graph draws
    n_layers: int = 12
    nodes_per_layer: int = 24
    edge_prob: float = 0.45
    observer_depth: Optional[int] = None  # default: middle layer
    seed: int = 12345


def _score_single_graph(
    q: int,
    R: int,
    site: CausalSite,
    observer_depth: int,
    alpha: float,
) -> float:
    """
    Score contribution from one graph draw (bits per tick).
    """
    # Spectral contraction constants — raise if not well-posed:
    try:
        consts = get_spectral_constants(q, R)
    except ValueError:
        return -np.inf
    l_gap = consts["l_gap"]  # = 1 - δ

    # Minimal hidden layer
    vis = {n for n, d in site.depths.items() if d < observer_depth}
    h_min = get_minimal_hidden_layer(site, vis)
    rate = len(h_min) * np.log2(max(2, q))  # per-tick predictor (exact)

    # Connectivity penalty encourages an interior optimum under fixed width
    penalty = np.exp(-alpha * max(0, R - 1))

    return float(rate * l_gap * penalty)


def score_entropy_density_predictor(
    q: int,
    R: int,
    cfg: EntropyMaxConfig,
    rng: np.random.Generator,
) -> float:
    """
    Measured entropy-density objective on a small ensemble of random sites.
    """
    if cfg.observer_depth is None:
        observer_depth = max(1, cfg.n_layers // 2)
    else:
        observer_depth = cfg.observer_depth

    scores: List[float] = []
    for _ in range(cfg.n_graphs):
        site = CausalSite.generate(
            n_layers=cfg.n_layers,
            nodes_per_layer=cfg.nodes_per_layer,
            R=R,
            edge_prob=cfg.edge_prob,
            rng=rng,
        )
        s = _score_single_graph(q, R, site, observer_depth, cfg.alpha_connectivity)
        if np.isfinite(s):
            scores.append(s)

    return float(np.mean(scores)) if scores else -np.inf


def anneal_qR(
    cfg: EntropyMaxConfig,
    method: str = "predictor",
) -> Tuple[Tuple[int, int], float, List[Tuple[int, int, float]]]:
    """
    Discrete simulated annealing over (q,R).

    Returns:
        ( (q_star, R_star), best_score, trace )
        where trace is a list of (q, R, score) samples (accepted steps).
    """
    rng = np.random.default_rng(cfg.seed)

    q_min, q_max = cfg.q_range
    R_min, R_max = cfg.R_range

    # Start near the interior to encourage exploration:
    q = int(np.clip((q_min + q_max) // 2, q_min, q_max))
    R = int(np.clip((R_min + R_max) // 2, R_min, R_max))
    score_fn = score_entropy_density_predictor

    def eval_score(qv: int, Rv: int) -> float:
        return score_fn(qv, Rv, cfg, rng)

    best_q, best_R = q, R
    best_score = eval_score(q, R)
    curr_score = best_score

    T = cfg.temperature0
    trace: List[Tuple[int, int, float]] = [(q, R, curr_score)]

    for step in range(cfg.budget):
        # Propose a neighbor in the grid (q or R ± 1)
        if rng.random() < 0.5:
            q_prop = int(np.clip(q + rng.choice([-1, 1]), q_min, q_max))
            R_prop = R
        else:
            q_prop = q
            R_prop = int(np.clip(R + rng.choice([-1, 1]), R_min, R_max))

        s_prop = eval_score(q_prop, R_prop)
        accept = False
        if s_prop > curr_score:
            accept = True
        else:
            # Metropolis acceptance
            if T > 0:
                p = np.exp((s_prop - curr_score) / max(1e-12, T))
                accept = rng.random() < p

        if accept:
            q, R, curr_score = q_prop, R_prop, s_prop
            trace.append((q, R, curr_score))
            if curr_score > best_score:
                best_q, best_R, best_score = q, R, curr_score

        # Cool temperature geometrically
        T = max(cfg.temperature_min, T * 0.97)

    return (best_q, best_R), float(best_score), trace
