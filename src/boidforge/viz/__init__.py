"""Visualization engine: deterministic replay of ``.bfs`` streams.

This subsystem reads precomputed frames and renders them. It comprises the
:class:`~boidforge.viz.replay.ReplayEngine` (drives playback), the
:class:`~boidforge.viz.renderer.Renderer` (ModernGL drawing), the
:class:`~boidforge.viz.camera.Camera` (center-of-mass follow, zoom/pan), the
interactive :class:`~boidforge.viz.app.InteractiveApp`, and the
:class:`~boidforge.viz.video.VideoExporter` (FFmpeg output).

It MUST NOT import :mod:`boidforge.solver` or :mod:`boidforge._native`, and it
never advances physics. Heavy GPU dependencies (ModernGL, pyglet) are imported
lazily inside the modules/methods that need them, so importing this package on a
headless node without a GL driver is safe.
"""

from __future__ import annotations

from boidforge.viz.camera import Camera
from boidforge.viz.renderer import ColorMode, RenderConfig, Renderer
from boidforge.viz.replay import ReplayConfig, ReplayEngine
from boidforge.viz.video import VideoExporter

__all__ = [
    "ReplayEngine",
    "ReplayConfig",
    "Camera",
    "Renderer",
    "RenderConfig",
    "ColorMode",
    "VideoExporter",
]
