"""Cross-backend equivalence: L1, L2, and L3 must produce identical results.

The naive backend (L1) is ground truth. These tests advance each backend from
the same seeded initial state and assert bit-identical buffers every step.
L2/L3 are expected to xfail until implemented; this is the correctness gate
referenced in ``CLAUDE.md`` §3. The L1 tests below pin the reference behaviour
(determinism, speed clamp, in-bounds, readable stream) that L2/L3 must match.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from boidforge.core.config import BoundaryMode, SimulationConfig
from boidforge.core.state import SimulationState
from boidforge.core.types import DTYPE
from boidforge.io.reader import FrameReader
from boidforge.solver import SOLVERS, NaiveSolver


def test_registry_contains_three_backends() -> None:
    """The solver registry exposes exactly the three required levels."""
    assert set(SOLVERS) == {"naive-l1", "spatial-hash-l2", "native-l3"}


def _advance(cfg: SimulationConfig) -> SimulationState:
    """Initialize and advance an L1 solver through ``cfg.steps``; return state."""
    solver = NaiveSolver(cfg)
    state = solver.initialize()
    for _ in range(cfg.steps):
        solver.step(state)
    return state


@pytest.mark.parametrize("boundary", [BoundaryMode.WRAP, BoundaryMode.REFLECT])
def test_l1_is_deterministic(boundary: BoundaryMode) -> None:
    """Two L1 runs with the same config are bit-identical (no wall-clock/RNG drift)."""
    cfg = SimulationConfig(n_boids=200, steps=30, seed=42, boundary=boundary)
    a = _advance(cfg)
    b = _advance(cfg)
    for comp in ("px", "py", "vx", "vy"):
        np.testing.assert_array_equal(getattr(a, comp), getattr(b, comp))
        assert getattr(a, comp).dtype == DTYPE


def test_l1_output_is_finite_and_speed_clamped() -> None:
    """L1 keeps speeds within [min_speed, max_speed] and never produces NaN/Inf."""
    cfg = SimulationConfig(n_boids=300, steps=40, seed=7)
    state = _advance(cfg)
    for comp in ("px", "py", "vx", "vy"):
        assert np.all(np.isfinite(getattr(state, comp)))
    speed = np.sqrt(state.vx**2 + state.vy**2)
    assert np.all(speed <= cfg.max_speed + 1e-2)
    assert np.all(speed >= cfg.min_speed - 1e-2)


@pytest.mark.parametrize("boundary", [BoundaryMode.WRAP, BoundaryMode.REFLECT])
def test_l1_positions_in_bounds(boundary: BoundaryMode) -> None:
    """Both boundary modes keep positions inside the world rectangle."""
    cfg = SimulationConfig(n_boids=200, steps=40, seed=3, boundary=boundary)
    state = _advance(cfg)
    assert np.all(state.px >= 0.0) and np.all(state.px <= cfg.world_width)
    assert np.all(state.py >= 0.0) and np.all(state.py <= cfg.world_height)


def test_l1_run_writes_readable_stream(tmp_path: Path) -> None:
    """run() streams to disk and the .bfs replays bit-identically to recompute."""
    cfg = SimulationConfig(n_boids=128, steps=15, seed=11)
    path = tmp_path / "l1.bfs"
    frames = NaiveSolver(cfg).run(path)
    assert frames == cfg.steps

    replay = NaiveSolver(cfg)
    state = replay.initialize()
    with FrameReader(path) as reader:
        assert reader.header.frame_count == cfg.steps
        for t in range(cfg.steps):
            replay.step(state)
            frame = reader.read_frame()
            assert frame is not None and frame.timestep == t
            np.testing.assert_array_equal(frame.state.px, state.px)
            np.testing.assert_array_equal(frame.state.vy, state.vy)
        assert reader.read_frame() is None


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
