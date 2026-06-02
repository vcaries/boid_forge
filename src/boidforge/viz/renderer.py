"""ModernGL renderer: instanced point sprites with velocity color and trails.

The renderer uploads per-frame SoA buffers to GPU vertex buffers and draws all
boids in a single instanced call. Velocity magnitude maps to color; an
accumulation framebuffer fades previous frames to produce motion trails.
Optional post-processing (bloom) is applied as a fullscreen pass.

ModernGL is imported lazily so the package imports without a GPU/driver present.
This module contains no simulation logic and never imports the solver.
"""

from __future__ import annotations

from dataclasses import dataclass

from boidforge.core.state import SimulationState
from boidforge.viz.camera import Camera


@dataclass(slots=True)
class RenderConfig:
    """Renderer appearance and quality settings.

    Attributes:
        width: Framebuffer width in pixels.
        height: Framebuffer height in pixels.
        point_size: Base sprite size in pixels.
        trail_decay: Per-frame trail fade in ``[0, 1]``; 0 disables trails.
        bloom: Whether to apply the bloom/glow post pass.
        background: RGBA clear color.
    """

    width: int = 1920
    height: int = 1080
    point_size: float = 3.0
    trail_decay: float = 0.92
    bloom: bool = True
    background: tuple[float, float, float, float] = (0.02, 0.02, 0.05, 1.0)


class Renderer:
    """Stateful ModernGL renderer for a single replay session."""

    def __init__(self, config: RenderConfig) -> None:
        """Create the GL context, shader programs, and framebuffers.

        Args:
            config: Appearance and quality settings.
        """
        raise NotImplementedError

    def draw(self, state: SimulationState, camera: Camera) -> None:
        """Render one frame of boids with the given camera transform.

        Args:
            state: Frame state to draw.
            camera: View transform for this frame.
        """
        raise NotImplementedError

    def read_pixels(self) -> bytes:
        """Read the current framebuffer as raw RGBA bytes.

        Used by the video exporter to capture a rendered frame.

        Returns:
            Tightly packed RGBA8 pixels, row-major, length ``width*height*4``.
        """
        raise NotImplementedError

    def release(self) -> None:
        """Release all GPU resources held by this renderer."""
        raise NotImplementedError
