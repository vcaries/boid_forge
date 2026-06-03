"""L1 — naive O(N²) Python solver (correctness reference).

This backend compares every boid against every other and is the *ground truth*:
L2 and L3 are validated against its output. It is vectorized per boid (an outer
Python loop over the ``N`` boids, with neighbour math vectorized across all
others), so it is correct and readable rather than fast.

Boid model (acceleration form), evaluated synchronously from the start-of-step
state for every boid:

* **Separation** ``a_sep = Σ_j (p_i - p_j) / |p_i - p_j|²`` over neighbours
  within ``r_sep`` — a soft inverse-distance push.
* **Alignment** ``a_ali = mean(v_j) - v_i`` over neighbours within ``r_ali``.
* **Cohesion** ``a_coh = mean(p_j) - p_i`` over neighbours within ``r_coh``.

Total acceleration ``a = w_sep·a_sep + w_ali·a_ali + w_coh·a_coh`` is clamped to
``max_force``; velocity is integrated and clamped to ``[min_speed, max_speed]``;
position is integrated by ``dt`` and the boundary mode applied.

Determinism contract: all constants are coerced to ``float32`` and every
neighbour reduction is a ``float32`` sum over the neighbour subset taken in
ascending boid-index order. L2/L3 reduce over the identical set in the identical
order, which is what makes the three backends bit-identical.
"""

from __future__ import annotations

import numpy as np

from boidforge.core.config import BoundaryMode
from boidforge.core.state import SimulationState
from boidforge.core.types import DTYPE
from boidforge.solver.base import Solver


class NaiveSolver(Solver):
    """All-pairs reference solver, ``O(N²)`` per timestep."""

    name = "naive-l1"

    def step(self, state: SimulationState) -> None:
        """Advance one timestep using exhaustive all-pairs neighbour search.

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
        index = np.arange(n)

        for i in range(n):
            dx = px - px[i]  # neighbour minus self
            dy = py - py[i]
            dist2 = dx * dx + dy * dy
            not_self = index != i

            ax = zero
            ay = zero

            sep = not_self & (dist2 < r_sep2)
            if np.any(sep):
                inv = one / dist2[sep]
                ax = ax + w_sep * np.sum(-dx[sep] * inv, dtype=DTYPE)
                ay = ay + w_sep * np.sum(-dy[sep] * inv, dtype=DTYPE)

            ali = not_self & (dist2 < r_ali2)
            if np.any(ali):
                ax = ax + w_ali * (np.mean(vx[ali], dtype=DTYPE) - vx[i])
                ay = ay + w_ali * (np.mean(vy[ali], dtype=DTYPE) - vy[i])

            coh = not_self & (dist2 < r_coh2)
            if np.any(coh):
                ax = ax + w_coh * (np.mean(px[coh], dtype=DTYPE) - px[i])
                ay = ay + w_coh * (np.mean(py[coh], dtype=DTYPE) - py[i])

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

    def _apply_boundary(self, state: SimulationState) -> None:
        """Apply the configured edge behaviour in place.

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
