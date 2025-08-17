# src/emergent/gravity_fp.py
"""
First-principles gravity witness using BD-style curvature.

Adds a block-level Einstein witness based on `geom_fp.bd_curvature_edge`
and your *existing* memory density definition so that the new witness aligns
numerically with the current pipeline (and can be tuned).

This module is **additive**; it does not change `gravity.py` (M11), which
remains as in your green test suite.
"""
from __future__ import annotations

from typing import Dict, Optional, Set, Tuple

import numpy as np

from emergent.geom_fp import bd_curvature_edge
from emergent.gravity import get_memory_density_edge  # reuse current definition
from emergent.poset import CausalSite, Edge, NodeID


def get_local_coupling_g_fp(
    site: CausalSite,
    edge: Edge,
    *,
    scheme: str = "4D-lite",
) -> float:
    """
    First-principles coupling g = kappa_bd / rho_mem on a single edge.

    Returns NaN if denominator is zero.
    """
    kappa = bd_curvature_edge(site, edge, scheme=scheme)
    rho = get_memory_density_edge(site, edge)  # same as current pipeline
    if rho == 0:
        return float("nan")
    return float(kappa / rho)


def analyze_gravity_in_block_fp(
    site: CausalSite,
    block_of_nodes: Set[NodeID],
    *,
    scheme: str = "4D-lite",
) -> Dict[str, float]:
    """
    Average first-principles coupling g over all edges fully contained in `block`.

    Returns
    -------
    dict with keys:
      - "avg_g_fp" : float (NaN if no valid edges)
      - "num_edges": int
    """
    edges = [
        (u, v)
        for u in block_of_nodes
        if u in site.adj
        for v in site.adj[u]
        if v in block_of_nodes
    ]
    if not edges:
        return {"avg_g_fp": float("nan"), "num_edges": 0}

    vals = []
    for e in edges:
        g = get_local_coupling_g_fp(site, e, scheme=scheme)
        if not (isinstance(g, float) and np.isnan(g)):
            vals.append(g)

    if not vals:
        return {"avg_g_fp": float("nan"), "num_edges": len(edges)}
    return {"avg_g_fp": float(np.mean(vals)), "num_edges": len(edges)}
