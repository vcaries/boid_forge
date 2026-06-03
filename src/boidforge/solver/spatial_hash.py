"""L3 — uniform-grid spatial-hash Python solver (~O(N)).

Same physics and same ``float32`` arithmetic as
:class:`~boidforge.solver.naive.NaiveSolver`, but neighbour search is reduced
from all-pairs ``O(N²)`` to ``~O(N)`` with a uniform grid. Boids are bucketed
into cells of side ``config.neighbor_radius`` (a plain ``dict`` keyed by integer
cell coordinates); each boid then examines only its own cell and the eight
adjacent ones — a 3×3 block.

That 3×3 block is sufficient because the cell side equals the neighbour radius:
if two boids are within that radius then ``|Δx|`` and ``|Δy|`` are each below the
cell side, so their integer cell coordinates differ by at most one on each axis.
Every boid passing any of the sep/ali/coh masks (all radii ``≤``
``neighbor_radius``) is therefore guaranteed to be among the gathered candidates;
the grid only prunes boids that could never be neighbours.

Bit-identity to L1 is preserved by reducing over the *same* neighbours in the
*same* order: candidates pulled from the nine cells are sorted by global boid
index before any reduction, so the compacted ``float32`` summation tree is the
one L1 (and L2) use. The grid changes *which* pairs are inspected, never *how*
the surviving neighbour contributions are summed. This is the determinism gate
in ``tests/test_solvers_equivalence.py``.

Cell binning is done in ``float64`` purely to choose buckets; it never feeds the
physics, so it cannot perturb the result — it only has to assign each boid to a
cell consistently.
"""

from __future__ import annotations

import numpy as np

from boidforge.core.config import BoundaryMode
from boidforge.core.state import SimulationState
from boidforge.core.types import DTYPE
from boidforge.solver.base import Solver


class SpatialHashSolver(Solver):
    """Uniform-grid accelerated solver, ~``O(N·k)`` per timestep."""

    name = "spatial-hash-l3"

    def step(self, state: SimulationState) -> None:
        """Advance one timestep using uniform-grid neighbour queries.

        Builds the grid from the start-of-step positions, then for each boid
        gathers candidates from its 3×3 cell block, sorts them by global index
        to match L1's reduction order, and applies the identical ``float32``
        sep/ali/coh reductions, force clamp, integration, and speed clamp.

        Args:
            state: The state to mutate in place.
        """
        cfg = self.config
        n = state.n
        px, py, vx, vy = state.px, state.py, state.vx, state.vy

        dt = np.float32(cfg.dt)
        r_sep2 = np.float32(cfg.r_sep * cfg.r_sep)
        r_ali2 = np.float32(cfg.r_ali * cfg.r_ali)
        r_coh2 = np.float32(cfg.r_coh * cfg.r_coh)
        w_sep = np.float32(cfg.w_sep)
        w_ali = np.float32(cfg.w_ali)
        w_coh = np.float32(cfg.w_coh)
        max_force = np.float32(cfg.max_force)
        max_force2 = np.float32(cfg.max_force * cfg.max_force)
        max_speed = np.float32(cfg.max_speed)
        min_speed = np.float32(cfg.min_speed)
        zero = np.float32(0.0)
        one = np.float32(1.0)

        new_vx = np.empty(n, dtype=DTYPE)
        new_vy = np.empty(n, dtype=DTYPE)

        # Bin boids into a uniform grid whose cell side is the neighbour radius.
        # float64 is used only to pick buckets and never enters the physics.
        s = float(cfg.neighbor_radius)
        cell_x: list[int] = np.floor(px.astype(np.float64) / s).astype(np.int64).tolist()
        cell_y: list[int] = np.floor(py.astype(np.float64) / s).astype(np.int64).tolist()
        grid = self._build_grid(cell_x, cell_y)

        for i in range(n):
            # Gather candidates from the 3×3 block, then sort by global index so
            # the surviving neighbours are reduced in L1's ascending-index order.
            cx = cell_x[i]
            cy = cell_y[i]
            cand_list: list[int] = []
            for gx in (cx - 1, cx, cx + 1):
                for gy in (cy - 1, cy, cy + 1):
                    bucket = grid.get((gx, gy))
                    if bucket is not None:
                        cand_list.extend(bucket)
            cand = np.asarray(cand_list, dtype=np.intp)
            cand.sort()

            pxc = px[cand]
            pyc = py[cand]
            vxc = vx[cand]
            vyc = vy[cand]
            dx = pxc - px[i]  # neighbour minus self
            dy = pyc - py[i]
            dist2 = dx * dx + dy * dy
            not_self = cand != i

            ax = zero
            ay = zero

            sep = not_self & (dist2 < r_sep2)
            if np.any(sep):
                inv = one / dist2[sep]
                ax = ax + w_sep * np.sum(-dx[sep] * inv, dtype=DTYPE)
                ay = ay + w_sep * np.sum(-dy[sep] * inv, dtype=DTYPE)

            ali = not_self & (dist2 < r_ali2)
            if np.any(ali):
                ax = ax + w_ali * (np.mean(vxc[ali], dtype=DTYPE) - vx[i])
                ay = ay + w_ali * (np.mean(vyc[ali], dtype=DTYPE) - vy[i])

            coh = not_self & (dist2 < r_coh2)
            if np.any(coh):
                ax = ax + w_coh * (np.mean(pxc[coh], dtype=DTYPE) - px[i])
                ay = ay + w_coh * (np.mean(pyc[coh], dtype=DTYPE) - py[i])

            a2 = ax * ax + ay * ay
            if a2 > max_force2:
                scale = max_force / np.sqrt(a2)
                ax = ax * scale
                ay = ay * scale

            nvx = vx[i] + ax * dt
            nvy = vy[i] + ay * dt

            speed = np.sqrt(nvx * nvx + nvy * nvy)
            if speed > max_speed:
                scale = max_speed / speed
                nvx = nvx * scale
                nvy = nvy * scale
            elif speed < min_speed and speed > zero:
                scale = min_speed / speed
                nvx = nvx * scale
                nvy = nvy * scale

            new_vx[i] = nvx
            new_vy[i] = nvy

        state.vx[:] = new_vx
        state.vy[:] = new_vy
        state.px += state.vx * dt
        state.py += state.vy * dt
        self._apply_boundary(state)

    def _build_grid(self, cell_x: list[int], cell_y: list[int]) -> dict[tuple[int, int], list[int]]:
        """Bucket boid indices into uniform grid cells.

        Indices are appended while iterating boids in ascending order, so every
        bucket — and any concatenation of buckets — is already index-sorted.

        Args:
            cell_x: Integer x cell coordinate per boid.
            cell_y: Integer y cell coordinate per boid.

        Returns:
            Mapping from a ``(cell_x, cell_y)`` key to the ascending list of
            boid indices that fall in that cell.
        """
        grid: dict[tuple[int, int], list[int]] = {}
        for i, (cx, cy) in enumerate(zip(cell_x, cell_y, strict=True)):
            grid.setdefault((cx, cy), []).append(i)
        return grid

    def _apply_boundary(self, state: SimulationState) -> None:
        """Apply the configured edge behaviour in place.

        Identical to the naive backend's boundary handling; duplicated here so
        the two solvers share no runtime state.

        Args:
            state: State whose positions (and, for reflection, velocities) are
                adjusted to respect the world rectangle.
        """
        cfg = self.config
        w = np.float32(cfg.world_width)
        h = np.float32(cfg.world_height)
        two = np.float32(2.0)

        if cfg.boundary is BoundaryMode.WRAP:
            np.mod(state.px, w, out=state.px)
            np.mod(state.py, h, out=state.py)
            return

        # REFLECT: mirror the position back inside and flip the normal velocity.
        left = state.px < np.float32(0.0)
        state.px[left] = -state.px[left]
        state.vx[left] = -state.vx[left]
        right = state.px > w
        state.px[right] = two * w - state.px[right]
        state.vx[right] = -state.vx[right]

        bottom = state.py < np.float32(0.0)
        state.py[bottom] = -state.py[bottom]
        state.vy[bottom] = -state.vy[bottom]
        top = state.py > h
        state.py[top] = two * h - state.py[top]
        state.vy[top] = -state.vy[top]
