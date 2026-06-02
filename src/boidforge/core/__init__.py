"""Core shared types: configuration, simulation state, and constants.

This subpackage holds the data structures shared by every subsystem. It depends
only on the standard library and NumPy and must not import :mod:`boidforge.solver`,
:mod:`boidforge.io`, or :mod:`boidforge.viz`.
"""

from __future__ import annotations

from boidforge.core.config import BoundaryMode, SimulationConfig
from boidforge.core.state import SimulationState
from boidforge.core.types import DIM, DTYPE, FloatArray

__all__ = [
    "BoundaryMode",
    "SimulationConfig",
    "SimulationState",
    "FloatArray",
    "DTYPE",
    "DIM",
]
