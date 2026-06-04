"""L4 — native CPython C-extension solver (main performance target).

This backend is a thin Python wrapper. It seeds state, then hands the four
contiguous ``float32`` SoA buffers to the C kernel in :mod:`boidforge._native`,
which performs the uniform-grid update in C with the GIL released and no
per-step allocation. The wrapper holds no physics itself; the algorithm lives in
``native/``. Output must match L1/L2/L3 bit-for-bit.
"""

from __future__ import annotations

from boidforge.core.config import BoundaryMode
from boidforge.core.state import SimulationState
from boidforge.solver.base import Solver

#: Map the boundary enum to the integer codes exposed by the C kernel.
_BOUNDARY_CODE: dict[BoundaryMode, int] = {
    BoundaryMode.WRAP: 0,
    BoundaryMode.REFLECT: 1,
}


class NativeSolver(Solver):
    """C-extension solver targeting 10k–100k boids, ~``O(N·k)`` per timestep.

    Seeding is inherited from :class:`~boidforge.solver.base.Solver` so the
    initial state is bit-identical to every other backend. Each step delegates
    to :func:`boidforge._native.step`, which performs the uniform-grid update in
    C with the GIL released and mutates the SoA buffers in place.
    """

    name = "native-l4"

    def step(self, state: SimulationState) -> None:
        """Advance one timestep by delegating to the C kernel.

        Passes ``px, py, vx, vy`` (C-contiguous ``float32``) plus the config
        parameters to :func:`boidforge._native.step`, which mutates the buffers
        in place. The import is deferred so the package still imports when the
        compiled extension is absent.

        Args:
            state: The state to mutate in place.

        Raises:
            ImportError: If the compiled ``boidforge._native`` extension is not
                built/importable.
        """
        from boidforge import _native

        cfg = self.config
        _native.step(
            state.px,
            state.py,
            state.vx,
            state.vy,
            dt=float(cfg.dt),
            r_sep=float(cfg.r_sep),
            r_ali=float(cfg.r_ali),
            r_coh=float(cfg.r_coh),
            w_sep=float(cfg.w_sep),
            w_ali=float(cfg.w_ali),
            w_coh=float(cfg.w_coh),
            max_speed=float(cfg.max_speed),
            min_speed=float(cfg.min_speed),
            max_force=float(cfg.max_force),
            world_w=float(cfg.world_width),
            world_h=float(cfg.world_height),
            boundary=_BOUNDARY_CODE[cfg.boundary],
        )
