"""Write→read round-trip for the ``.bfs`` stream.

These tests define the contract the writer and reader must satisfy: frames read
back must equal frames written, in order, with identical SoA payloads.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

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


def test_header_frame_count_backpatched(tmp_path: Path) -> None:
    """On close the global header records the number of frames written."""
    cfg = SimulationConfig(n_boids=32, steps=7)
    rng = np.random.default_rng(cfg.seed)
    path = tmp_path / "count.bfs"
    with FrameWriter(path, cfg) as writer:
        for t in range(cfg.steps):
            writer.write(t, _random_state(cfg.n_boids, rng))

    with FrameReader(path) as reader:
        assert reader.header.frame_count == cfg.steps
        assert reader.header.max_boids == cfg.n_boids
        assert reader.header.seed == cfg.seed


def test_seek_frame_matches_sequential(tmp_path: Path) -> None:
    """seek_frame(i) lands on the same payload as sequential reading."""
    cfg = SimulationConfig(n_boids=48, steps=6)
    rng = np.random.default_rng(cfg.seed)
    written = [_random_state(cfg.n_boids, rng) for _ in range(cfg.steps)]
    path = tmp_path / "seek.bfs"
    with FrameWriter(path, cfg) as writer:
        for t, state in enumerate(written):
            writer.write(t, state)

    with FrameReader(path) as reader:
        reader.seek_frame(4)
        frame = reader.read_frame()
        assert frame is not None
        assert frame.timestep == 4
        np.testing.assert_array_equal(frame.state.px, written[4].px)
        np.testing.assert_array_equal(frame.state.vy, written[4].vy)
