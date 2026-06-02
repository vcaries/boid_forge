"""Tests for the concrete core data structures (config + SoA state)."""

from __future__ import annotations

import numpy as np
import pytest

from boidforge.core.config import BoundaryMode, SimulationConfig
from boidforge.core.state import SimulationState
from boidforge.core.types import DTYPE


def test_config_roundtrip() -> None:
    """to_dict/from_dict preserves a config including the boundary enum."""
    cfg = SimulationConfig(n_boids=128, steps=10, boundary=BoundaryMode.REFLECT)
    restored = SimulationConfig.from_dict(cfg.to_dict())
    assert restored == cfg
    assert restored.boundary is BoundaryMode.REFLECT


def test_config_validation() -> None:
    """Invalid configurations are rejected at construction."""
    with pytest.raises(ValueError):
        SimulationConfig(n_boids=0, steps=10)
    with pytest.raises(ValueError):
        SimulationConfig(n_boids=10, steps=10, min_speed=5.0, max_speed=5.0)


def test_neighbor_radius_is_max() -> None:
    """neighbor_radius reports the largest interaction radius."""
    cfg = SimulationConfig(n_boids=4, steps=1, r_sep=5.0, r_ali=20.0, r_coh=15.0)
    assert cfg.neighbor_radius == 20.0


def test_state_allocate_invariants() -> None:
    """Allocated state is float32, C-contiguous, and equal length."""
    state = SimulationState.allocate(64)
    assert state.n == 64
    for arr in (state.px, state.py, state.vx, state.vy):
        assert arr.shape == (64,)
        assert arr.dtype == DTYPE
        assert arr.flags["C_CONTIGUOUS"]


def test_state_rejects_mismatched_arrays() -> None:
    """Constructing state with mismatched arrays raises."""
    with pytest.raises(ValueError):
        SimulationState(
            px=np.zeros(4, dtype=DTYPE),
            py=np.zeros(5, dtype=DTYPE),
            vx=np.zeros(4, dtype=DTYPE),
            vy=np.zeros(4, dtype=DTYPE),
        )
