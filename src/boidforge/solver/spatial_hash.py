"""L3 — uniform-grid spatial-hash Python solver (~O(N)).

Boids are bucketed into a uniform grid whose cell size equals
``config.neighbor_radius``, so each boid only examines its own and the eight
adjacent cells. Results must match :class:`~boidforge.solver.naive.NaiveSolver`
bit-for-bit; to guarantee that, candidate neighbors are sorted by boid index
before accumulation so summation order matches the naive backend.
"""

from __future__ import annotations

from boidforge.core.state import SimulationState
from boidforge.solver.base import Solver


class SpatialHashSolver(Solver):
    """Uniform-grid accelerated solver, ~``O(N·k)`` per timestep."""

    name = "spatial-hash-l3"

    def initialize(self) -> SimulationState:
        """Seed initial state identically to the reference backend.

        Returns:
            A seeded :class:`SimulationState` of ``config.n_boids`` boids.
        """
        raise NotImplementedError

    def step(self, state: SimulationState) -> None:
        """Advance one timestep using uniform-grid neighbor queries.

        Args:
            state: The state to mutate in place.
        """
        raise NotImplementedError

    def _build_grid(self, state: SimulationState) -> object:
        """Bucket boids into uniform grid cells for this timestep.

        Args:
            state: Current state to index.

        Returns:
            An internal grid structure mapping cell coordinates to boid indices.
        """
        raise NotImplementedError
