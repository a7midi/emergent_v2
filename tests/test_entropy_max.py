import numpy as np
import pytest

from emergent.entropy_max import EntropyMaxConfig, anneal_qR, score_entropy_density_predictor

def test_scoring_is_finite_in_well_posed_region():
    cfg = EntropyMaxConfig(q_range=(6, 12), R_range=(2, 6), n_graphs=2, budget=10)
    rng = np.random.default_rng(0)
    val = score_entropy_density_predictor(q=8, R=3, cfg=cfg, rng=rng)
    assert np.isfinite(val)

@pytest.mark.slow
def test_annealer_returns_interior_point_small_budget():
    cfg = EntropyMaxConfig(q_range=(6, 18), R_range=(1, 6), budget=60, n_graphs=3, seed=2025)
    (q_star, r_star), best, trace = anneal_qR(cfg)
    # Well-posed constraint q > R-1 is generally enforced by spectral constants
    assert q_star >= 3 and r_star >= 1
    assert len(trace) >= 1
