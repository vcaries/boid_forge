"""FFmpeg video export.

Streams rendered RGBA frames to an FFmpeg subprocess via a pipe to encode an
H.264 (or other) video. Keeps no frames in memory beyond the current one. This
is an output sink only; it neither reads ``.bfs`` nor runs physics.
"""

from __future__ import annotations

import os
from types import TracebackType


class VideoExporter:
    """Pipes rendered frames to FFmpeg to produce a video file.

    Intended use as a context manager::

        with VideoExporter(path, 1920, 1080, fps=60) as v:
            for frame in reader:
                renderer.draw(frame.state, camera)
                v.write_frame(renderer.read_pixels())
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        width: int,
        height: int,
        fps: int = 60,
        crf: int = 18,
    ) -> None:
        """Spawn an FFmpeg process configured to read raw RGBA from stdin.

        Args:
            path: Output video path (e.g. ``.mp4``).
            width: Frame width in pixels.
            height: Frame height in pixels.
            fps: Output frame rate.
            crf: x264 constant rate factor (lower = higher quality).
        """
        raise NotImplementedError

    def write_frame(self, rgba: bytes) -> None:
        """Write one raw RGBA8 frame to the encoder.

        Args:
            rgba: Row-major RGBA8 pixels, length ``width*height*4``.
        """
        raise NotImplementedError

    def close(self) -> None:
        """Flush and close FFmpeg's stdin, then wait for encoding to finish."""
        raise NotImplementedError

    def __enter__(self) -> VideoExporter:
        """Enter the context manager.

        Returns:
            This exporter.
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Finalize encoding on context exit."""
        self.close()
