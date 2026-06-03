"""2D camera: center-of-mass follow, zoom, and pan.

The camera maps world coordinates to normalized device coordinates for the
renderer. It can track the boid flock's center of mass with optional smoothing
and supports interactive zoom/pan. This module is pure math and has no GPU or
solver dependency.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from boidforge.core.state import SimulationState

#: A 4x4 transform flattened in column-major order (OpenGL convention).
Mat4 = tuple[
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
]  # noqa: E501


@dataclass(slots=True)
class Camera:
    """View transform for rendering a frame.

    The camera defines an orthographic window onto the world. At ``zoom == 1``
    exactly one world unit maps to one framebuffer pixel, so the default world
    (1920x1080) fills a 1080p framebuffer. Larger ``zoom`` magnifies.

    Attributes:
        center_x: World-space X of the view center.
        center_y: World-space Y of the view center.
        zoom: Scale factor; larger zooms in. Always kept positive.
        follow: Whether to track the flock center of mass each frame.
        smoothing: Exponential follow smoothing in ``[0, 1)``; 0 = instant snap,
            values near 1 trail the flock slowly.
    """

    center_x: float = 0.0
    center_y: float = 0.0
    zoom: float = 1.0
    follow: bool = True
    smoothing: float = 0.85

    def update(self, state: SimulationState) -> None:
        """Advance the camera toward the flock center of mass.

        When :attr:`follow` is enabled the view center is moved toward the mean
        boid position with exponential smoothing controlled by
        :attr:`smoothing`. With follow disabled the center is left untouched so
        manual panning is preserved.

        Args:
            state: Current frame state used to compute the center of mass.
        """
        if not self.follow or state.n == 0:
            return
        com_x = float(np.mean(state.px))
        com_y = float(np.mean(state.py))
        alpha = 1.0 - _clamp(self.smoothing, 0.0, 0.999)
        self.center_x += (com_x - self.center_x) * alpha
        self.center_y += (com_y - self.center_y) * alpha

    def view_matrix(self, viewport_w: int, viewport_h: int) -> Mat4:
        """Build the world->clip transform for the current view.

        Produces an orthographic projection centered on
        ``(center_x, center_y)`` whose visible extent is the viewport size
        divided by :attr:`zoom`. The matrix is returned flattened in
        column-major order so it can be written straight to a GLSL ``mat4``
        uniform.

        Args:
            viewport_w: Framebuffer width in pixels.
            viewport_h: Framebuffer height in pixels.

        Returns:
            A 4x4 column-major transform as a flat tuple of 16 floats.
        """
        zoom = max(self.zoom, 1e-6)
        half_w = (viewport_w * 0.5) / zoom
        half_h = (viewport_h * 0.5) / zoom
        left = self.center_x - half_w
        right = self.center_x + half_w
        bottom = self.center_y - half_h
        top = self.center_y + half_h
        return _ortho(left, right, bottom, top)

    def frame_world(self, width: float, height: float, viewport_w: int, viewport_h: int) -> None:
        """Center on a world rectangle and choose a zoom that fits it.

        Convenience used when first opening a stream so the whole simulation is
        visible regardless of world size.

        Args:
            width: World width to fit.
            height: World height to fit.
            viewport_w: Framebuffer width in pixels.
            viewport_h: Framebuffer height in pixels.
        """
        self.center_x = width * 0.5
        self.center_y = height * 0.5
        if width <= 0 or height <= 0:
            return
        self.zoom = min(viewport_w / width, viewport_h / height)

    def pan(self, dx: float, dy: float) -> None:
        """Translate the view center by a world-space delta.

        Args:
            dx: World-space X translation.
            dy: World-space Y translation.
        """
        self.center_x += dx
        self.center_y += dy

    def pan_pixels(self, dx_px: float, dy_px: float) -> None:
        """Pan by a screen-pixel delta, converting through the current zoom.

        Args:
            dx_px: Pixel delta along X (e.g. mouse drag).
            dy_px: Pixel delta along Y.
        """
        zoom = max(self.zoom, 1e-6)
        self.center_x -= dx_px / zoom
        self.center_y -= dy_px / zoom

    def zoom_by(self, factor: float) -> None:
        """Multiply the zoom level.

        Args:
            factor: Multiplicative zoom change (>1 zooms in). Non-positive
                factors are ignored.
        """
        if factor <= 0.0:
            return
        self.zoom = _clamp(self.zoom * factor, 1e-4, 1e6)


def _clamp(value: float, lo: float, hi: float) -> float:
    """Clamp ``value`` into ``[lo, hi]``."""
    return max(lo, min(hi, value))


def _ortho(left: float, right: float, bottom: float, top: float) -> Mat4:
    """Build a column-major orthographic matrix mapping the rect to clip space.

    Maps ``[left, right] x [bottom, top]`` to the OpenGL clip cube with a fixed
    near/far of ``[-1, 1]``. Returned column-major (OpenGL memory order).

    Args:
        left: Left edge in world units.
        right: Right edge in world units.
        bottom: Bottom edge in world units.
        top: Top edge in world units.

    Returns:
        The 4x4 projection as a flat 16-float tuple.
    """
    rl = right - left or 1e-6
    tb = top - bottom or 1e-6
    sx = 2.0 / rl
    sy = 2.0 / tb
    tx = -(right + left) / rl
    ty = -(top + bottom) / tb
    # Column-major: columns are contiguous.
    return (
        sx,
        0.0,
        0.0,
        0.0,
        0.0,
        sy,
        0.0,
        0.0,
        0.0,
        0.0,
        -1.0,
        0.0,
        tx,
        ty,
        0.0,
        1.0,
    )
