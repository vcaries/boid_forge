"""L3 — native CPython C-extension solver (main performance target).

This backend is a thin Python wrapper. It seeds state, then hands the four
contiguous ``float32`` SoA buffers to the C kernel in :mod:`boidforge._native`,
which performs the uniform-grid update in C with the GIL released and no
per-step allocation. The wrapper holds no physics itself; the algorithm lives in
``native/``. Output must match L1/L2 bit-for-bit.
"""

from __future__ import annotations

from boidforge.core.state import SimulationState
from boidforge.solver.base import Solver


class NativeSolver(Solver):
    """C-extension solver targeting 10k–100k boids, ~``O(N·k)`` per timestep."""

    name = "native-l3"

    def initialize(self) -> SimulationState:
        """Seed initial state identically to the reference backend.

        Returns:
            A seeded :class:`SimulationState` of ``config.n_boids`` boids.
        """
        raise NotImplementedError

    def step(self, state: SimulationState) -> None:
        """Advance one timestep by delegating to the C kernel.

        Passes ``px, py, vx, vy`` (C-contiguous ``float32``) plus the config
        parameters to :func:`boidforge._native.step`, which mutates the buffers
        in place.

        Args:
            state: The state to mutate in place.
        """
        raise NotImplementedError
