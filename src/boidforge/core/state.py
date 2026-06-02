"""Struct-of-Arrays simulation state.

The entire simulation operates on four parallel, C-contiguous ``float32``
arrays — positions ``(px, py)`` and velocities ``(vx, vy)``. There are no per
boid objects and no array-of-structs anywhere in the hot path. This layout is
cache-friendly for the solver, maps directly to the component-major ``.bfs``
payload (zero repacking on write), and is what the C kernel consumes via the
buffer protocol.

State construction (buffer allocation) is concrete here; *evolving* the state is
the solver's job and lives in :mod:`boidforge.solver`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from boidforge.core.types import DTYPE, FloatArray


@dataclass(slots=True)
class SimulationState:
    """Mutable SoA buffers for ``N`` boids.

    The four arrays are always the same length ``N``, ``float32``, and
    C-contiguous. Backends mutate them in place each timestep.

    Attributes:
        px: X positions, shape ``(N,)``.
        py: Y positions, shape ``(N,)``.
        vx: X velocities, shape ``(N,)``.
        vy: Y velocities, shape ``(N,)``.
    """

    px: FloatArray
    py: FloatArray
    vx: FloatArray
    vy: FloatArray

    def __post_init__(self) -> None:
        """Enforce the SoA invariants every consumer relies on.

        Raises:
            ValueError: If arrays differ in length, dtype, or contiguity.
        """
        arrays = (self.px, self.py, self.vx, self.vy)
        n = self.px.shape[0]
        for arr in arrays:
            if arr.shape != (n,):
                raise ValueError("all state arrays must be 1-D of equal length")
            if arr.dtype != DTYPE:
                raise ValueError(f"state arrays must be {DTYPE}")
            if not arr.flags["C_CONTIGUOUS"]:
                raise ValueError("state arrays must be C-contiguous")

    @property
    def n(self) -> int:
        """Number of boids ``N`` currently held.

        Returns:
            The common length of the four component arrays.
        """
        return int(self.px.shape[0])

    @classmethod
    def allocate(cls, n: int) -> SimulationState:
        """Allocate zero-initialized state for ``n`` boids.

        Args:
            n: Number of boids to allocate.

        Returns:
            A :class:`SimulationState` with four zeroed ``float32`` arrays.

        Raises:
            ValueError: If ``n`` is not positive.
        """
        if n <= 0:
            raise ValueError("n must be positive")
        return cls(
            px=np.zeros(n, dtype=DTYPE),
            py=np.zeros(n, dtype=DTYPE),
            vx=np.zeros(n, dtype=DTYPE),
            vy=np.zeros(n, dtype=DTYPE),
        )

    def copy(self) -> SimulationState:
        """Return a deep copy of the state buffers.

        Returns:
            A new :class:`SimulationState` owning independent arrays.
        """
        return SimulationState(
            px=self.px.copy(),
            py=self.py.copy(),
            vx=self.vx.copy(),
            vy=self.vy.copy(),
        )
