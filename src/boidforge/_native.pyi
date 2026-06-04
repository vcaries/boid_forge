"""Type stub for the compiled ``boidforge._native`` C extension (L4 kernel).

The extension is built from ``native/`` via CMake/scikit-build-core. This stub
describes its surface so ``mypy --strict`` can check the wrapper without the
binary being present.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

#: Integer code for toroidal (wrap) boundary handling.
BOUNDARY_WRAP: int

#: Integer code for reflecting (hard-wall) boundary handling.
BOUNDARY_REFLECT: int

def step(
    px: npt.NDArray[np.float32],
    py: npt.NDArray[np.float32],
    vx: npt.NDArray[np.float32],
    vy: npt.NDArray[np.float32],
    *,
    dt: float,
    r_sep: float,
    r_ali: float,
    r_coh: float,
    w_sep: float,
    w_ali: float,
    w_coh: float,
    max_speed: float,
    min_speed: float,
    max_force: float,
    world_w: float,
    world_h: float,
    boundary: int,
) -> None:
    """Advance one timestep in place over the four float32 SoA buffers."""
