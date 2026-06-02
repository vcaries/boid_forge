"""BoidForge: decoupled high-performance 2D Boids solver and replay engine.

The package is split into two subsystems that communicate only through the
on-disk ``.bfs`` binary stream (see :mod:`boidforge.io` and
``docs/architecture.md``):

* **Compute layer** (:mod:`boidforge.solver`, :mod:`boidforge.core`,
  :mod:`boidforge.io`) advances boid physics and writes snapshots.
* **Post-processing layer** (:mod:`boidforge.viz`) reads snapshots and renders.

Importing :mod:`boidforge` pulls in only the lightweight compute/core symbols.
Visualization dependencies (ModernGL, pyglet) are imported lazily from
:mod:`boidforge.viz` so headless compute environments never require a GPU stack.
"""

from __future__ import annotations

from boidforge.core.config import SimulationConfig
from boidforge.core.state import SimulationState

__version__ = "0.1.0"

__all__ = ["SimulationConfig", "SimulationState", "__version__"]
