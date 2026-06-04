"""Replay engine: drive playback of a ``.bfs`` stream through the renderer.

The engine is the visualization entry point. It opens a stream with
:class:`~boidforge.io.reader.FrameReader`, decodes frames, and either launches an
interactive tunable window (:class:`~boidforge.viz.app.InteractiveApp`) or
renders the sequence offscreen to a video via
:class:`~boidforge.viz.video.VideoExporter`. It performs no simulation.

Frames are decoded into memory so the interactive player can scrub freely and so
the camera can be auto-framed from the data (the ``.bfs`` header records no world
extent). For very large streams use :attr:`ReplayConfig.max_frames` to cap it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from boidforge.io.reader import Frame, FrameReader
from boidforge.viz.renderer import ColorMode, RenderConfig


@dataclass(slots=True)
class ReplayConfig:
    """Playback settings.

    Attributes:
        fps: Target playback / export frame rate.
        loop: Whether to restart at end of stream (interactive mode).
        export_path: If set, render offscreen and encode to this video path.
        width: Render width in pixels.
        height: Render height in pixels.
        colormap: Initial colormap name.
        color_mode: Initial colour mapping (``speed``/``heading``/``uniform``).
        crf: x264 quality for export (lower = better).
        max_frames: Optional cap on frames decoded (0 = all).
        auto_speed: Auto-calibrate the colour speed reference from the data.
    """

    fps: int = 60
    loop: bool = True
    export_path: str | None = None
    width: int = 1600
    height: int = 900
    colormap: str = "turbo"
    color_mode: str = "speed"
    crf: int = 18
    max_frames: int = 0
    auto_speed: bool = True
    render: RenderConfig | None = field(default=None)


class ReplayEngine:
    """Sequential replay of a ``.bfs`` stream to screen or video."""

    def __init__(
        self,
        stream_path: str | os.PathLike[str],
        replay: ReplayConfig,
    ) -> None:
        """Open the stream, decode frames, and prepare render settings.

        Args:
            stream_path: Source ``.bfs`` file to replay.
            replay: Playback configuration.

        Raises:
            ValueError: If the stream contains no frames.
        """
        self._replay = replay
        self._frames = self._load(stream_path, replay.max_frames)
        if not self._frames:
            raise ValueError(f"{os.fspath(stream_path)} contains no frames")
        self._world = self._world_extent(self._frames)
        self._render = replay.render or self._make_render_config(replay)

    @staticmethod
    def _load(path: str | os.PathLike[str], max_frames: int) -> list[Frame]:
        """Decode frames from the stream into memory (optionally capped)."""
        frames: list[Frame] = []
        with FrameReader(path) as reader:
            for frame in reader:
                frames.append(frame)
                if max_frames and len(frames) >= max_frames:
                    break
        return frames

    @staticmethod
    def _world_extent(frames: list[Frame]) -> tuple[float, float]:
        """Estimate world ``(width, height)`` from observed boid positions."""
        max_x = max(float(f.state.px.max()) for f in frames)
        max_y = max(float(f.state.py.max()) for f in frames)
        return (max(max_x, 1.0), max(max_y, 1.0))

    def _speed_range(self) -> tuple[float, float]:
        """The 5th/95th-percentile boid speed across a sample of frames.

        Mapping the colour ramp onto this observed range (rather than onto an
        absolute ``speed/max`` ratio) is what makes slow boids read cold and
        fast boids hot: tightly clamped flocks still get the full spread of
        colour instead of collapsing to a single hue.
        """
        sample = self._frames[:: max(1, len(self._frames) // 24)]
        speeds = [
            np.sqrt(f.state.vx.astype(np.float64) ** 2 + f.state.vy.astype(np.float64) ** 2)
            for f in sample
        ]
        allv = np.concatenate(speeds)
        if not allv.size:
            return (0.0, 1.0)
        lo = float(np.percentile(allv, 5.0))
        hi = float(np.percentile(allv, 95.0))
        if hi - lo < 1e-3:
            hi = lo + 1.0
        return (lo, hi)

    def _make_render_config(self, replay: ReplayConfig) -> RenderConfig:
        """Build a :class:`RenderConfig` from the replay settings + data stats."""
        cfg = RenderConfig(
            width=replay.width,
            height=replay.height,
            colormap=replay.colormap,
            color_mode=ColorMode[replay.color_mode.upper()],
        )
        if replay.auto_speed:
            cfg.speed_lo, cfg.speed_hi = self._speed_range()
        # A density cell of roughly the neighbour spacing gives a readable
        # crowding field; tie it to the world so it scales with the simulation.
        cfg.density_cell = max(min(self._world) / 60.0, 1.0)
        return cfg

    def run(self) -> None:
        """Play the stream interactively, or render it to video if exporting.

        When :attr:`ReplayConfig.export_path` is set, frames are encoded to a
        video file offscreen; otherwise an interactive window opens.
        """
        if self._replay.export_path:
            self._run_export(self._replay.export_path)
        else:
            self._run_interactive()

    def _run_interactive(self) -> None:
        """Launch the windowed, live-tunable player."""
        from boidforge.viz.app import InteractiveApp

        app = InteractiveApp(
            self._frames,
            self._render,
            self._world,
            fps=self._replay.fps,
            loop=self._replay.loop,
        )
        app.run()

    def _run_export(self, path: str) -> None:
        """Render every frame offscreen and encode it to ``path``."""
        from boidforge.viz.camera import Camera
        from boidforge.viz.renderer import Renderer
        from boidforge.viz.video import VideoExporter

        max_n = max(f.state.n for f in self._frames)
        renderer = Renderer(self._render, ctx=None, max_boids=max_n)
        camera = Camera()
        camera.frame_world(self._world[0], self._world[1], self._render.width, self._render.height)
        try:
            with VideoExporter(
                path,
                self._render.width,
                self._render.height,
                fps=self._replay.fps,
                crf=self._replay.crf,
            ) as video:
                for frame in self._frames:
                    camera.update(frame.state)
                    renderer.draw(frame.state, camera, present=False)
                    video.write_frame(renderer.read_pixels())
        finally:
            renderer.release()
