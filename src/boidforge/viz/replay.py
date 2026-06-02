"""Replay engine: drive playback of a ``.bfs`` stream through the renderer.

The engine is the visualization entry point. It opens a stream with
:class:`~boidforge.io.reader.FrameReader`, reads frames sequentially, updates the
:class:`~boidforge.viz.camera.Camera`, and submits each frame to the
:class:`~boidforge.viz.renderer.Renderer` — optionally capturing to a
:class:`~boidforge.viz.video.VideoExporter`. It performs no simulation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(slots=True)
class ReplayConfig:
    """Playback settings.

    Attributes:
        fps: Target playback / export frame rate.
        loop: Whether to restart at end of stream (interactive mode).
        export_path: If set, render offscreen and encode to this video path.
    """

    fps: int = 60
    loop: bool = False
    export_path: str | None = None


class ReplayEngine:
    """Sequential replay of a ``.bfs`` stream to screen or video."""

    def __init__(
        self,
        stream_path: str | os.PathLike[str],
        replay: ReplayConfig,
    ) -> None:
        """Open the stream and prepare the renderer and camera.

        Args:
            stream_path: Source ``.bfs`` file to replay.
            replay: Playback configuration.
        """
        raise NotImplementedError

    def run(self) -> None:
        """Play the stream to completion (or until the window is closed).

        Reads frames in order, advancing the camera and rendering each. When
        :attr:`ReplayConfig.export_path` is set, frames are encoded to video
        instead of shown interactively.
        """
        raise NotImplementedError
