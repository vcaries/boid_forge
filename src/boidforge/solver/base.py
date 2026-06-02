"""Abstract solver interface and the record-to-disk run loop.

Every backend subclasses :class:`Solver` and implements two methods:
:meth:`Solver.initialize` (seed the SoA state deterministically from the config)
and :meth:`Solver.step` (advance the state one timestep in place). The concrete
:meth:`Solver.run` wires a solver to a :class:`~boidforge.io.writer.FrameWriter`
— this is the entire compute pipeline and contains no rendering.
"""

from __future__ import annotations

import abc
import os

from boidforge.core.config import SimulationConfig
from boidforge.core.state import SimulationState
from boidforge.io.writer import FrameWriter


class Solver(abc.ABC):
    """Base class for all boid simulation backends.

    Subclasses set the class attribute :attr:`name` and implement
    :meth:`initialize` and :meth:`step`. The three boid rules, integration,
    speed clamping, and boundary handling are defined by the config and must be
    applied identically across backends to satisfy the determinism contract.

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

    @abc.abstractmethod
    def initialize(self) -> SimulationState:
        """Create the seeded initial state.

        Positions and velocities are a pure function of ``config.seed`` and
        ``config.n_boids`` so that every backend starts identically.

        Returns:
            A freshly allocated, seeded :class:`SimulationState`.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def step(self, state: SimulationState) -> None:
        """Advance ``state`` by one timestep, in place.

        Applies separation, alignment, and cohesion, clamps speed, integrates
        position by ``config.dt``, and applies the boundary mode.

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
