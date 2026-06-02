"""Immutable simulation configuration.

:class:`SimulationConfig` is the single source of every parameter that affects
results. Determinism requires that backends read values from here rather than
embedding literals, and that the RNG be seeded solely from :attr:`SimulationConfig.seed`.
The object is frozen so a configuration cannot drift mid-run.
"""

from __future__ import annotations

import enum
from dataclasses import asdict, dataclass
from typing import Any


class BoundaryMode(enum.Enum):
    """How boids behave at the edges of the world rectangle.

    Attributes:
        WRAP: Toroidal world; a boid leaving one edge re-enters the opposite.
        REFLECT: Hard walls; the velocity component normal to the wall flips.
    """

    WRAP = "wrap"
    REFLECT = "reflect"


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    """Complete, immutable description of a simulation run.

    All fields that influence numerical output live here so that a run is fully
    reproducible from ``(SimulationConfig, seed)`` alone.

    Attributes:
        n_boids: Number of boids ``N`` in the simulation.
        steps: Number of timesteps to advance and record.
        dt: Integration timestep (seconds).
        world_width: Width of the world rectangle in world units.
        world_height: Height of the world rectangle in world units.
        boundary: Edge behavior, see :class:`BoundaryMode`.
        r_sep: Separation radius; neighbors closer than this are avoided.
        r_ali: Alignment radius; neighbors within steer heading.
        r_coh: Cohesion radius; neighbors within attract position.
        w_sep: Separation weight.
        w_ali: Alignment weight.
        w_coh: Cohesion weight.
        max_speed: Upper speed clamp (world units / second).
        min_speed: Lower speed clamp (world units / second).
        max_force: Per-step steering acceleration clamp.
        seed: RNG seed; the sole source of initial-state randomness.
    """

    n_boids: int
    steps: int
    dt: float = 0.05
    world_width: float = 1920.0
    world_height: float = 1080.0
    boundary: BoundaryMode = BoundaryMode.WRAP

    r_sep: float = 12.0
    r_ali: float = 30.0
    r_coh: float = 30.0

    w_sep: float = 1.5
    w_ali: float = 1.0
    w_coh: float = 1.0

    max_speed: float = 180.0
    min_speed: float = 40.0
    max_force: float = 220.0

    seed: int = 0

    def __post_init__(self) -> None:
        """Validate invariants that the solvers and format rely on.

        Raises:
            ValueError: If any field is outside its permitted range.
        """
        if self.n_boids <= 0:
            raise ValueError("n_boids must be positive")
        if self.steps <= 0:
            raise ValueError("steps must be positive")
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")
        if self.world_width <= 0.0 or self.world_height <= 0.0:
            raise ValueError("world dimensions must be positive")
        if min(self.r_sep, self.r_ali, self.r_coh) <= 0.0:
            raise ValueError("interaction radii must be positive")
        if self.min_speed < 0.0 or self.max_speed <= self.min_speed:
            raise ValueError("require 0 <= min_speed < max_speed")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")

    @property
    def neighbor_radius(self) -> float:
        """Largest interaction radius; sets the spatial-grid cell size.

        Returns:
            The maximum of the separation, alignment, and cohesion radii.
        """
        return max(self.r_sep, self.r_ali, self.r_coh)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dict (enums become their values).

        Returns:
            A plain dictionary suitable for JSON/CSV provenance records.
        """
        data = asdict(self)
        data["boundary"] = self.boundary.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SimulationConfig:
        """Construct from a dict produced by :meth:`to_dict`.

        Args:
            data: Mapping of field names to values; ``boundary`` may be a string.

        Returns:
            A validated :class:`SimulationConfig`.
        """
        payload = dict(data)
        boundary = payload.get("boundary", BoundaryMode.WRAP)
        if isinstance(boundary, str):
            payload["boundary"] = BoundaryMode(boundary)
        return cls(**payload)
