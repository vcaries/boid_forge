"""L1 — naive O(N²) Python solver (correctness reference).

This backend compares every boid against every other. It is the slowest and the
*ground truth*: L2 and L3 are validated against its output. Neighbor
contributions are summed in ascending boid-index order to fix floating-point
accumulation order across backends (see the determinism contract).
"""

from __future__ import annotations

from boidforge.core.state import SimulationState
from boidforge.solver.base import Solver


class NaiveSolver(Solver):
    """All-pairs reference solver, ``O(N²)`` per timestep."""

    name = "naive-l1"

    def initialize(self) -> SimulationState:
        """Seed initial state deterministically from the config.

        Returns:
            A seeded :class:`SimulationState` of ``config.n_boids`` boids.
        """
        raise NotImplementedError

    def step(self, state: SimulationState) -> None:
        """Advance one timestep using exhaustive all-pairs neighbor search.

        Args:
            state: The state to mutate in place.
        """
        raise NotImplementedError
