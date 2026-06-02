"""Visualization engine: deterministic replay of ``.bfs`` streams.

This subsystem reads precomputed frames and renders them. It comprises the
:class:`~boidforge.viz.replay.ReplayEngine` (drives playback), the
:class:`~boidforge.viz.renderer.Renderer` (ModernGL drawing), the
:class:`~boidforge.viz.camera.Camera` (center-of-mass follow, zoom/pan), and the
:class:`~boidforge.viz.video.VideoExporter` (FFmpeg output).

It MUST NOT import :mod:`boidforge.solver` or :mod:`boidforge._native`, and it
never advances physics. Heavy GPU dependencies (ModernGL, pyglet) are imported
lazily inside the modules that need them.
"""

from __future__ import annotations

from boidforge.viz.camera import Camera
from boidforge.viz.replay import ReplayConfig, ReplayEngine

__all__ = ["ReplayEngine", "ReplayConfig", "Camera"]
