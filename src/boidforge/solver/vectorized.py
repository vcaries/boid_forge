"""L2 — vectorized NumPy O(N²) solver (bit-identical to the L1 reference).

This backend is a faithful re-expression of
:class:`~boidforge.solver.naive.NaiveSolver`: same physics, same ``float32``
arithmetic, same neighbour-iteration order — but written to lean on NumPy
instead of Python. The all-pairs geometry is built once per step as broadcast
``N×N`` matrices, the separation contributions are precomputed in bulk, and the
force/speed clamps plus integration run as whole-array operations rather than a
per-boid Python loop.

It is **not** a different algorithm. Each boid's neighbour reductions are still
``np.sum``/``np.mean`` over the *same* compacted neighbour subset in the *same*
ascending boid-index order as L1, so the ``float32`` pairwise-summation tree is
identical and the output is bit-for-bit equal to L1 — the determinism gate in
``tests/test_solvers_equivalence.py``. The one irreducible Python loop that
remains is the per-boid reduction: a fully matrix-reduced sum would group the
``float32`` additions differently (pairwise over a zero-padded row instead of
the compacted neighbour list) and break bit-identity.

Trade-off: speed for memory. The pairwise matrices cost ``O(N²)`` memory
(several ``N×N`` arrays) versus L1's ``O(N)``. This backend targets the same
boid-count range as L1; the ~``O(N)`` memory/time backends are L3 (spatial
hash) and L4 (native).
"""

from __future__ import annotations

import numpy as np

from boidforge.core.config import BoundaryMode
from boidforge.core.state import SimulationState
from boidforge.core.types import DTYPE
from boidforge.solver.base import Solver


class VectorizedSolver(Solver):
    """Vectorized all-pairs solver: ``O(N²)`` time, bit-identical to L1."""

    name = "vectorized-l2"

    def step(self, state: SimulationState) -> None:
        """Advance one timestep using broadcast all-pairs math.

        Builds the pairwise displacement, distance, and separation-contribution
        matrices once, loops per boid only for the neighbour reductions (whose
        ordering the determinism contract pins), then finishes with whole-array
        force clamping, integration, and speed clamping.

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

        # Pairwise geometry, built once. dx[i, j] = px[j] - px[i] ("neighbour
        # minus self"), matching the naive backend's per-boid ``px - px[i]``.
        dx = px[np.newaxis, :] - px[:, np.newaxis]
        dy = py[np.newaxis, :] - py[:, np.newaxis]
        dist2 = dx * dx + dy * dy
        not_self = ~np.eye(n, dtype=bool)

        sep_mask = not_self & (dist2 < r_sep2)
        ali_mask = not_self & (dist2 < r_ali2)
        coh_mask = not_self & (dist2 < r_coh2)

        # Separation contribution for each ordered pair: -(p_j - p_i) / dist².
        # Computed as reciprocal-then-multiply to match L1's ``inv = 1/dist2;
        # -dx * inv`` exactly (true division ``-dx / dist2`` would round
        # differently). Self pairs give inf/nan here but are never gathered
        # (the mask excludes self); a coincident non-self pair reproduces L1's
        # nan identically.
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = one / dist2
            sx = -dx * inv
            sy = -dy * inv

        ax = np.empty(n, dtype=DTYPE)
        ay = np.empty(n, dtype=DTYPE)
        for i in range(n):
            sep = sep_mask[i]
            ali = ali_mask[i]
            coh = coh_mask[i]

            axi = zero
            ayi = zero

            if sep.any():
                axi = axi + w_sep * np.sum(sx[i][sep], dtype=DTYPE)
                ayi = ayi + w_sep * np.sum(sy[i][sep], dtype=DTYPE)

            if ali.any():
                axi = axi + w_ali * (np.mean(vx[ali], dtype=DTYPE) - vx[i])
                ayi = ayi + w_ali * (np.mean(vy[ali], dtype=DTYPE) - vy[i])

            if coh.any():
                axi = axi + w_coh * (np.mean(px[coh], dtype=DTYPE) - px[i])
                ayi = ayi + w_coh * (np.mean(py[coh], dtype=DTYPE) - py[i])

            ax[i] = axi
            ay[i] = ayi

        # Force clamp, integration, and speed clamp as whole-array ops. Each is
        # the per-element equivalent of the naive backend's scalar branch: where
        # a clamp does not apply, the scale is exactly 1.0 and ``v * 1.0 == v``.
        a2 = ax * ax + ay * ay
        over_force = a2 > max_force2
        with np.errstate(divide="ignore", invalid="ignore"):
            f_scale = np.where(over_force, max_force / np.sqrt(a2), one)
        ax = ax * f_scale
        ay = ay * f_scale

        new_vx = vx + ax * dt
        new_vy = vy + ay * dt

        speed = np.sqrt(new_vx * new_vx + new_vy * new_vy)
        over_speed = speed > max_speed
        under_speed = (speed < min_speed) & (speed > zero)
        with np.errstate(divide="ignore", invalid="ignore"):
            s_scale = np.where(
                over_speed,
                max_speed / speed,
                np.where(under_speed, min_speed / speed, one),
            )
        new_vx = new_vx * s_scale
        new_vy = new_vy * s_scale

        state.vx[:] = new_vx
        state.vy[:] = new_vy
        state.px += state.vx * dt
        state.py += state.vy * dt
        self._apply_boundary(state)

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
