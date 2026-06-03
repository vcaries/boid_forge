"""Shared numeric types and invariants.

Centralizing the element dtype and spatial dimension here guarantees that the
solver, the binary format, and the visualizer all agree on the in-memory and
on-disk representation. Changing :data:`DTYPE` is a format-breaking change and
requires bumping ``FORMAT_VERSION`` in :mod:`boidforge.io.format`.
"""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
import numpy.typing as npt

#: Element type for all boid state and on-disk payloads. IEEE-754 single
#: precision is the format and determinism contract; do not widen casually.
DTYPE: np.dtype[np.float32] = np.dtype(np.float32)

#: Spatial dimensionality of the simulation (2D).
DIM: int = 2

#: A C-contiguous 1-D ``float32`` array — the canonical SoA component buffer.
FloatArray: TypeAlias = npt.NDArray[np.float32]
