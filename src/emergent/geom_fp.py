# src/emergent/geom_fp.py
"""
First-principles geometric diagnostics (Alexandrov interval chain counting).

This module provides *interval-level* path/chain counting inside I+(u) ∩ I-(v)
and a Benincasa–Dowker–style curvature estimator for an edge (u -> v).  It does
NOT modify the proxy tools in `geom.py`; tests for those remain green
(see tests for M08 in your suite).  This adds a tunable, first-principles
counterpart you can iterate on.

Key ideas
---------
- Interval I(u,v) := I+(u) ∩ I-(v) using your existing metric helpers.
- Chain counts C_k := number of directed k-edge chains fully contained in I(u,v).
- BD-style series (4D-lite, tunable): kappa_bd(u->v) ≍ α0*N0 + α1*C1 + α2*C2 + α3*C3,
  where N0=|I|, C1=edges, C2=2-chains (paths with 2 edges), C3=3-chains, etc.
  The default coefficients encode an alternating-sign, short-range stencil that
  is robust on finite DAGs and easy to tune empirically.  Replace with the exact
  coefficients from your papers when you finalize them.

API
---
- `alexandrov_interval(site, u, v) -> set[int]`
- `count_k_chains(site, nodes, k) -> int`
- `bd_curvature_edge(site, edge, scheme='4D-lite', coeffs=None) -> float`

This module is additive and does not change geom.py (M08).
"""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

from emergent.metric import get_alexandrov_interval
from emergent.poset import CausalSite, Edge, NodeID


def alexandrov_interval(site: CausalSite, u: NodeID, v: NodeID) -> Set[NodeID]:
    """Inclusive Alexandrov interval I+(u) ∩ I-(v)."""
    return get_alexandrov_interval(site, u, v)


def _subgraph_adjacency(site: CausalSite, nodes: Set[NodeID]) -> Dict[NodeID, List[NodeID]]:
    """Adjacency restricted to `nodes`."""
    sub_adj: Dict[NodeID, List[NodeID]] = {}
    for a in nodes:
        succs = site.adj.get(a, [])
        sub_adj[a] = [b for b in succs if b in nodes]
    return sub_adj


def count_k_chains(site: CausalSite, nodes: Set[NodeID], k: int) -> int:
    """
    Count directed k-edge chains fully contained in the induced subgraph on `nodes`.

    Examples:
      k=1 -> number of edges inside the subgraph
      k=2 -> number of length-2 paths (a->b->c), etc.

    Implementation: dynamic programming on a DAG; O(|E| * k).
    """
    if k <= 0:
        return 0
    if not nodes:
        return 0

    sub_adj = _subgraph_adjacency(site, nodes)

    # For k=1, just count edges:
    if k == 1:
        return sum(len(succs) for succs in sub_adj.values())

    # DP: ways[v][t] = number of t-edge paths starting at v
    # We fill from t=1..k and sum over starts for t==k
    ways_prev: Dict[NodeID, int] = {v: len(sub_adj[v]) for v in nodes}  # t=1
    if k == 1:
        return int(np.sum(list(ways_prev.values())))

    for t in range(2, k + 1):
        ways_curr: Dict[NodeID, int] = {}
        for v in nodes:
            w = 0
            for w1 in sub_adj[v]:
                w += ways_prev.get(w1, 0)
            ways_curr[v] = w
        ways_prev = ways_curr

    # total number of k-edge chains = sum over starts of ways_prev[start]
    return int(np.sum(list(ways_prev.values())))


def bd_curvature_edge(
    site: CausalSite,
    edge: Edge,
    *,
    scheme: str = "4D-lite",
    coeffs: Optional[Dict[str, float]] = None,
    max_chain_length: int = 3,
) -> float:
    """
    Benincasa–Dowker–style interval curvature estimator on a directed edge.

    We compute chain counts inside the inclusive interval I(u,v) and combine
    them with a short-range stencil (alternating sign) intended to mimic the
    BD curvature expansion on finite DAGs.

    Parameters
    ----------
    site : CausalSite
    edge : tuple[int,int]    # (u,v) must be an existing arrow
    scheme : {"4D-lite", "custom"}
        "4D-lite" uses default coefficients (safe & monotone).  Use "custom"
        with `coeffs` to inject your exact set once finalized.
    coeffs : dict[str,float] or None
        Keys among {"N0","C1","C2","C3"}; others ignored.
    max_chain_length : int
        How far to count chains (default 3, adequate for speed).

    Returns
    -------
    float : curvature estimate kappa_bd(u->v)
    """
    u, v = edge
    if v not in site.adj.get(u, []):
        raise ValueError(f"Edge {edge} not present in site.")

    interval = alexandrov_interval(site, u, v)
    if not interval:
        return 0.0

    N0 = float(len(interval))
    C: Dict[int, float] = {}
    for k in range(1, max_chain_length + 1):
        C[k] = float(count_k_chains(site, interval, k))

    # Default short-range alternating coefficients (robust on finite DAGs).
    # Replace with the exact BD 4D constants later if desired.
    if coeffs is None:
        if scheme == "4D-lite":
            # Motivated by alternating-sign, short-range stencil:
            # kappa ≈ -N0 + 3*C1 - 3*C2 + 1*C3  (tunable)
            coeffs = {"N0": -1.0, "C1": 3.0, "C2": -3.0, "C3": 1.0}
        else:
            coeffs = {"N0": -1.0, "C1": 1.0, "C2": 0.0, "C3": 0.0}

    kappa = coeffs.get("N0", 0.0) * N0
    for k, val in C.items():
        kappa += coeffs.get(f"C{k}", 0.0) * val
    return float(kappa)
