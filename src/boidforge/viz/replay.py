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
import numpy.typing as npt

from boidforge.io.reader import Frame, FrameReader
from boidforge.viz.renderer import ColorMode, RenderConfig


@dataclass(slots=True)
class ReplayConfig:
    """Playback, appearance, and camera settings for replay/export.

    The appearance overrides mirror the live control panel one-for-one. Each is
    ``None`` by default, meaning "use the computed default / auto-calibration";
    set one to pin that knob to an exact value so an exported video reproduces
    what was dialled in interactively.

    Attributes:
        fps: Target playback / export frame rate.
        loop: Whether to restart at end of stream (interactive mode).
        export_path: If set, render offscreen and encode to this video path.
        width: Render width in pixels.
        height: Render height in pixels.
        colormap: Colormap name.
        color_mode: Colour mapping (``speed``/``heading``/``uniform``/``density``).
        crf: x264 quality for export (lower = better).
        start_frame: Index of the first frame to replay/export (0 = beginning).
        max_frames: Optional cap on frames decoded from ``start_frame`` (0 = all).
        auto_speed: Auto-calibrate the colour speed range from the data (unless
            ``speed_lo``/``speed_hi`` are given explicitly).
        point_size: Base sprite diameter in pixels.
        glow: Sprite halo strength.
        intensity: Overall emission multiplier.
        trail_decay: Per-frame trail retention in ``[0, 1]``; 0 disables trails.
        bloom: Whether to apply the bloom pass.
        bloom_strength: Bloom add-back strength.
        bloom_threshold: Luminance threshold feeding the bloom.
        exposure: HDR exposure before tonemapping.
        vignette: Corner darkening in ``[0, 1]``.
        speed_lo: Speed mapped to the cold end of the ramp (overrides auto).
        speed_hi: Speed mapped to the hot end of the ramp (overrides auto).
        density_cell: Grid cell side for the DENSITY colour mode.
        uniform_t: Colour-map coordinate for the UNIFORM colour mode.
        zoom: Multiplier on the fit-to-world camera zoom (1.0 = whole world).
        follow: Whether the camera tracks the flock centre of mass.
        render: Fully-built RenderConfig override (bypasses the builder).
    """

    fps: int = 60
    loop: bool = True
    export_path: str | None = None
    width: int = 1600
    height: int = 900
    colormap: str = "turbo"
    color_mode: str = "speed"
    crf: int = 18
    start_frame: int = 0
    max_frames: int = 0
    auto_speed: bool = True

    point_size: float | None = None
    glow: float | None = None
    intensity: float | None = None
    trail_decay: float | None = None
    bloom: bool | None = None
    bloom_strength: float | None = None
    bloom_threshold: float | None = None
    exposure: float | None = None
    vignette: float | None = None
    speed_lo: float | None = None
    speed_hi: float | None = None
    density_cell: float | None = None
    uniform_t: float | None = None

    zoom: float = 1.0
    follow: bool = True

    render: RenderConfig | None = field(default=None)


class ReplayEngine:
    """Sequential replay of a ``.bfs`` stream to screen or video."""

    def __init__(
        self,
        stream_path: str | os.PathLike[str],
        replay: ReplayConfig,
    ) -> None:
        """Bind the engine to a stream and its playback settings.

        Frames are *not* decoded here: interactive playback loads them into
        memory (it needs random access for scrubbing), while export streams
        them one at a time so a multi-gigabyte stream does not have to fit in
        RAM. See :meth:`run`.

        Args:
            stream_path: Source ``.bfs`` file to replay.
            replay: Playback configuration.
        """
        self._path = stream_path
        self._replay = replay

    @staticmethod
    def _seek_reader(reader: FrameReader, start: int) -> None:
        """Position ``reader`` at frame ``start`` (fast seek, else skip-read).

        Uses the fixed-stride :meth:`FrameReader.seek_frame` when the stream
        records a ``max_boids`` bound (this package's writer always does), and
        falls back to reading and discarding for variable-length streams.

        Args:
            reader: An open reader positioned at the first frame.
            start: Zero-based frame index to advance to.
        """
        if start <= 0:
            return
        try:
            reader.seek_frame(start)
        except NotImplementedError:
            for _ in range(start):
                if reader.read_frame() is None:
                    break

    @staticmethod
    def _load(path: str | os.PathLike[str], start: int, max_frames: int) -> list[Frame]:
        """Decode frames ``[start, start+max_frames)`` from the stream into memory."""
        frames: list[Frame] = []
        with FrameReader(path) as reader:
            ReplayEngine._seek_reader(reader, start)
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

    @staticmethod
    def _speed_range(frames: list[Frame]) -> tuple[float, float]:
        """The 5th/95th-percentile boid speed across a sample of ``frames``.

        Mapping the colour ramp onto this observed range (rather than onto an
        absolute ``speed/max`` ratio) is what makes slow boids read cold and
        fast boids hot: tightly clamped flocks still get the full spread of
        colour instead of collapsing to a single hue.
        """
        sample = frames[:: max(1, len(frames) // 24)]
        speeds = [
            np.sqrt(f.state.vx.astype(np.float64) ** 2 + f.state.vy.astype(np.float64) ** 2)
            for f in sample
        ]
        return ReplayEngine._percentile_range(np.concatenate(speeds) if speeds else np.empty(0))

    @staticmethod
    def _percentile_range(speeds: npt.NDArray[np.float64]) -> tuple[float, float]:
        """5th/95th-percentile of a flat speed array, with a non-zero span."""
        if not speeds.size:
            return (0.0, 1.0)
        lo = float(np.percentile(speeds, 5.0))
        hi = float(np.percentile(speeds, 95.0))
        if hi - lo < 1e-3:
            hi = lo + 1.0
        return (lo, hi)

    def _scan(self) -> tuple[tuple[float, float], tuple[float, float], int, int]:
        """Single streaming pass for export: world, speed range, max N, count.

        Reads the stream once without holding it in memory, accumulating the
        world extent, a sampled speed distribution (stride chosen from the
        header frame count), the largest boid count, and the number of frames.

        Returns:
            ``(world, (speed_lo, speed_hi), max_boids, n_frames)``.
        """
        start = self._replay.start_frame
        cap = self._replay.max_frames
        with FrameReader(self._path) as reader:
            total = reader.header.frame_count
            if total > 0:
                total = max(0, total - start)
            if cap:
                total = cap if total < 0 else min(total, cap)
            stride = max(1, total // 24) if total > 0 else 1

            self._seek_reader(reader, start)
            max_x = max_y = 1.0
            max_n = 0
            count = 0
            sampled: list[npt.NDArray[np.float64]] = []
            for i, frame in enumerate(reader):
                if cap and i >= cap:
                    break
                st = frame.state
                max_x = max(max_x, float(st.px.max()))
                max_y = max(max_y, float(st.py.max()))
                max_n = max(max_n, st.n)
                # Cap the speed sample (matters when the header lacks a frame
                # count, so stride is 1 and we'd otherwise sample every frame).
                if i % stride == 0 and len(sampled) < 32:
                    sampled.append(
                        np.sqrt(st.vx.astype(np.float64) ** 2 + st.vy.astype(np.float64) ** 2)
                    )
                count += 1

        world = (max_x, max_y)
        srange = self._percentile_range(np.concatenate(sampled) if sampled else np.empty(0))
        return world, srange, max_n, count

    def _make_render_config(
        self,
        world: tuple[float, float],
        speed_range: tuple[float, float],
    ) -> RenderConfig:
        """Build a :class:`RenderConfig` from settings, data stats, and overrides.

        Order: start from the defaults, auto-calibrate the speed range and a
        data-scaled density cell, then apply any explicit appearance overrides
        from the replay config so a flag always wins over the auto value.

        Args:
            world: ``(width, height)`` world extent from the data.
            speed_range: ``(lo, hi)`` speed percentiles from the data.

        Returns:
            The resolved render configuration.
        """
        replay = self._replay
        cfg = RenderConfig(
            width=replay.width,
            height=replay.height,
            colormap=replay.colormap,
            color_mode=ColorMode[replay.color_mode.upper()],
        )
        if replay.auto_speed:
            cfg.speed_lo, cfg.speed_hi = speed_range
        # A density cell of roughly the neighbour spacing gives a readable
        # crowding field; tie it to the world so it scales with the simulation.
        cfg.density_cell = max(min(world) / 60.0, 1.0)

        # Apply explicit appearance overrides (each None unless the user set it).
        for field_name in (
            "point_size",
            "glow",
            "intensity",
            "trail_decay",
            "bloom",
            "bloom_strength",
            "bloom_threshold",
            "exposure",
            "vignette",
            "speed_lo",
            "speed_hi",
            "density_cell",
            "uniform_t",
        ):
            value = getattr(replay, field_name)
            if value is not None:
                setattr(cfg, field_name, value)
        return cfg

    def run(self) -> None:
        """Play the stream interactively, or render it to video if exporting.

        When :attr:`ReplayConfig.export_path` is set, frames are streamed and
        encoded to a video file offscreen (low memory); otherwise the stream is
        loaded into memory and an interactive window opens.

        Raises:
            ValueError: If the stream contains no frames.
        """
        if self._replay.export_path:
            self._run_export(self._replay.export_path)
        else:
            self._run_interactive()

    def _run_interactive(self) -> None:
        """Load the stream into memory and launch the live-tunable player."""
        from boidforge.viz.app import InteractiveApp

        frames = self._load(self._path, self._replay.start_frame, self._replay.max_frames)
        if not frames:
            raise ValueError(
                f"{os.fspath(self._path)} has no frames at start_frame={self._replay.start_frame}"
            )
        world = self._world_extent(frames)
        render = self._replay.render or self._make_render_config(world, self._speed_range(frames))
        app = InteractiveApp(
            frames,
            render,
            world,
            fps=self._replay.fps,
            loop=self._replay.loop,
            zoom=self._replay.zoom,
            follow=self._replay.follow,
        )
        app.run()

    def _run_export(self, path: str) -> None:
        """Stream every frame from disk, render offscreen, and encode to ``path``.

        Frames are read one at a time so a multi-gigabyte stream never has to
        be resident in memory — only the current frame plus the renderer's GPU
        buffers. The stream is scanned once up front for the world extent,
        speed range, and buffer sizing, then re-read to render.

        Args:
            path: Destination video file.

        Raises:
            ValueError: If the stream contains no frames.
        """
        from boidforge.viz.camera import Camera
        from boidforge.viz.renderer import Renderer
        from boidforge.viz.video import VideoExporter

        world, srange, max_n, n_frames = self._scan()
        if n_frames == 0:
            raise ValueError(
                f"{os.fspath(self._path)} has no frames at start_frame={self._replay.start_frame}"
            )
        render = self._replay.render or self._make_render_config(world, srange)

        renderer = Renderer(render, ctx=None, max_boids=max_n)
        camera = Camera()
        camera.frame_world(world[0], world[1], render.width, render.height)
        camera.zoom *= self._replay.zoom
        camera.follow = self._replay.follow
        cap = self._replay.max_frames
        try:
            with (
                VideoExporter(
                    path, render.width, render.height, fps=self._replay.fps, crf=self._replay.crf
                ) as video,
                FrameReader(self._path) as reader,
            ):
                self._seek_reader(reader, self._replay.start_frame)
                for i, frame in enumerate(reader):
                    if cap and i >= cap:
                        break
                    camera.update(frame.state)
                    renderer.draw(frame.state, camera, present=False)
                    video.write_frame(renderer.read_pixels())
        finally:
            renderer.release()
