"""2D camera: center-of-mass follow, zoom, and pan.

The camera maps world coordinates to normalized device coordinates for the
renderer. It can track the boid flock's center of mass with optional smoothing
and supports interactive zoom/pan. This module is pure math and has no GPU or
solver dependency.
"""

from __future__ import annotations

from dataclasses import dataclass

from boidforge.core.state import SimulationState


@dataclass(slots=True)
class Camera:
    """View transform for rendering a frame.

    Attributes:
        center_x: World-space X of the view center.
        center_y: World-space Y of the view center.
        zoom: Scale factor; larger zooms in.
        follow: Whether to track the flock center of mass each frame.
        smoothing: Exponential follow smoothing in ``[0, 1)``; 0 = instant.
    """

    center_x: float = 0.0
    center_y: float = 0.0
    zoom: float = 1.0
    follow: bool = True
    smoothing: float = 0.85

    def update(self, state: SimulationState) -> None:
        """Advance the camera toward the flock center of mass.

        Args:
            state: Current frame state used to compute the center of mass.
        """
        raise NotImplementedError

    def view_matrix(self, viewport_w: int, viewport_h: int) -> object:
        """Build the world→clip transform for the current view.

        Args:
            viewport_w: Framebuffer width in pixels.
            viewport_h: Framebuffer height in pixels.

        Returns:
            A 4×4 column-major transform (as a flat float sequence) for the GPU.
        """
        raise NotImplementedError

    def pan(self, dx: float, dy: float) -> None:
        """Translate the view center by a world-space delta.

        Args:
            dx: World-space X translation.
            dy: World-space Y translation.
        """
        raise NotImplementedError

    def zoom_by(self, factor: float) -> None:
        """Multiply the zoom level.

        Args:
            factor: Multiplicative zoom change (>1 zooms in).
        """
        raise NotImplementedError
