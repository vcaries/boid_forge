"""Write→read round-trip for the ``.bfs`` stream (pending implementation).

These tests define the contract the writer and reader must satisfy: frames read
back must equal frames written, in order, with identical SoA payloads. They are
expected to xfail until :mod:`boidforge.io` is implemented.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from boidforge.core.config import SimulationConfig
from boidforge.core.state import SimulationState
from boidforge.core.types import DTYPE
from boidforge.io.reader import FrameReader
from boidforge.io.writer import FrameWriter


def _random_state(n: int, rng: np.random.Generator) -> SimulationState:
    """Build a random SoA state for round-trip testing."""
    return SimulationState(
        px=rng.random(n, dtype=np.float32).astype(DTYPE),
        py=rng.random(n, dtype=np.float32).astype(DTYPE),
        vx=rng.random(n, dtype=np.float32).astype(DTYPE),
        vy=rng.random(n, dtype=np.float32).astype(DTYPE),
    )


@pytest.mark.xfail(raises=NotImplementedError, strict=True, reason="skeleton")
def test_write_read_roundtrip(tmp_path: Path) -> None:
    """Frames read back match frames written, byte-for-byte."""
    cfg = SimulationConfig(n_boids=256, steps=5)
    rng = np.random.default_rng(cfg.seed)
    written = [_random_state(cfg.n_boids, rng) for _ in range(cfg.steps)]

    path = tmp_path / "roundtrip.bfs"
    with FrameWriter(path, cfg) as writer:
        for t, state in enumerate(written):
            writer.write(t, state)

    with FrameReader(path) as reader:
        for t, expected in enumerate(written):
            frame = reader.read_frame()
            assert frame is not None
            assert frame.timestep == t
            np.testing.assert_array_equal(frame.state.px, expected.px)
            np.testing.assert_array_equal(frame.state.py, expected.py)
            np.testing.assert_array_equal(frame.state.vx, expected.vx)
            np.testing.assert_array_equal(frame.state.vy, expected.vy)
        assert reader.read_frame() is None
