"""Abstract solver interface, deterministic seeding, and the run loop.

Every backend subclasses :class:`Solver` and implements :meth:`Solver.step`
(advance the SoA state one timestep in place). Initial-state seeding is provided
*concretely* by :meth:`Solver.initialize` so that all backends start from a
bit-identical state derived solely from ``config.seed`` — a precondition of the
cross-level determinism contract. The concrete :meth:`Solver.run` wires a solver
to a :class:`~boidforge.io.writer.FrameWriter`; it contains no rendering.
"""

from __future__ import annotations

import abc
import os

import numpy as np

from boidforge.core.config import SimulationConfig
from boidforge.core.state import SimulationState
from boidforge.core.types import DTYPE
from boidforge.io.writer import FrameWriter


class Solver(abc.ABC):
    """Base class for all boid simulation backends.

    Subclasses set the class attribute :attr:`name` and implement :meth:`step`.
    The three boid rules, integration, speed clamping, and boundary handling are
    defined by the config and must be applied identically across backends to
    satisfy the determinism contract.

    Attributes:
        name: Stable backend identifier (e.g. ``"naive-l1"``).
        config: The immutable run configuration.
    """

    name: str = "base"

    def __init__(self, config: SimulationConfig) -> None:
        """Bind the solver to a configuration.

        Args:
            config: Immutable run configuration shared by all backends.
        """
        self.config = config

    def initialize(self) -> SimulationState:
        """Create the seeded initial state (shared by every backend).

        Positions are drawn uniformly over the world rectangle; velocities are
        drawn with uniform heading and uniform speed in ``[min_speed,
        max_speed]``. All draws come from a PCG64 generator seeded only by
        ``config.seed`` and in a fixed order, so the initial state is identical
        across L1/L2/L3 and across machines. Buffers are cast to ``float32``.

        Returns:
            A freshly allocated, seeded :class:`SimulationState`.
        """
        cfg = self.config
        rng = np.random.default_rng(cfg.seed)
        px = rng.uniform(0.0, cfg.world_width, cfg.n_boids).astype(DTYPE)
        py = rng.uniform(0.0, cfg.world_height, cfg.n_boids).astype(DTYPE)
        heading = rng.uniform(0.0, 2.0 * np.pi, cfg.n_boids)
        speed = rng.uniform(cfg.min_speed, cfg.max_speed, cfg.n_boids)
        vx = (speed * np.cos(heading)).astype(DTYPE)
        vy = (speed * np.sin(heading)).astype(DTYPE)
        return SimulationState(px=px, py=py, vx=vx, vy=vy)

    @abc.abstractmethod
    def step(self, state: SimulationState) -> None:
        """Advance ``state`` by one timestep, in place.

        Applies separation, alignment, and cohesion, clamps steering force,
        integrates velocity then position by ``config.dt``, and applies the
        boundary mode.

        Args:
            state: The state to mutate.
        """
        raise NotImplementedError

    def run(self, path: str | os.PathLike[str]) -> int:
        """Run the full simulation and stream snapshots to ``path``.

        Orchestration only: initialize once, then for each timestep advance the
        state and append a frame. Contains no physics and no rendering.

        Args:
            path: Destination ``.bfs`` file.

        Returns:
            The number of frames written (equal to ``config.steps``).
        """
        state = self.initialize()
        with FrameWriter(path, self.config) as writer:
            for t in range(self.config.steps):
                self.step(state)
                writer.write(t, state)
            return writer.frames_written
