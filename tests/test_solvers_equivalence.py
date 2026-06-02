"""Cross-backend equivalence: L1, L2, and L3 must produce identical results.

The naive backend (L1) is ground truth. These tests advance each backend from
the same seeded initial state and assert bit-identical buffers every step.
Expected to xfail until the solvers are implemented; this is the correctness
gate referenced in ``CLAUDE.md`` §3.
"""

from __future__ import annotations

import numpy as np
import pytest

from boidforge.core.config import SimulationConfig
from boidforge.solver import SOLVERS, NaiveSolver


def test_registry_contains_three_backends() -> None:
    """The solver registry exposes exactly the three required levels."""
    assert set(SOLVERS) == {"naive-l1", "spatial-hash-l2", "native-l3"}


@pytest.mark.parametrize("backend", ["spatial-hash-l2", "native-l3"])
@pytest.mark.parametrize("n_boids", [64, 512])
@pytest.mark.xfail(raises=NotImplementedError, strict=True, reason="skeleton")
def test_backend_matches_naive(backend: str, n_boids: int) -> None:
    """An accelerated backend matches the naive reference bit-for-bit."""
    cfg = SimulationConfig(n_boids=n_boids, steps=20, seed=1234)

    ref = NaiveSolver(cfg)
    other = SOLVERS[backend](cfg)
    ref_state = ref.initialize()
    other_state = other.initialize()

    for _ in range(cfg.steps):
        ref.step(ref_state)
        other.step(other_state)
        np.testing.assert_array_equal(ref_state.px, other_state.px)
        np.testing.assert_array_equal(ref_state.py, other_state.py)
        np.testing.assert_array_equal(ref_state.vx, other_state.vx)
        np.testing.assert_array_equal(ref_state.vy, other_state.vy)
