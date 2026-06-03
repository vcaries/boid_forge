"""FFmpeg video export.

Streams rendered RGBA frames to an FFmpeg subprocess via a pipe to encode an
H.264 (or other) video. Keeps no frames in memory beyond the current one. This
is an output sink only; it neither reads ``.bfs`` nor runs physics.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from types import TracebackType


class FFmpegNotFoundError(RuntimeError):
    """Raised when no ``ffmpeg`` executable is available on the system PATH."""


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
        ffmpeg: str | None = None,
    ) -> None:
        """Spawn an FFmpeg process configured to read raw RGBA from stdin.

        Args:
            path: Output video path (e.g. ``.mp4``).
            width: Frame width in pixels.
            height: Frame height in pixels.
            fps: Output frame rate.
            crf: x264 constant rate factor (lower = higher quality).
            ffmpeg: Explicit path to the ffmpeg binary; falls back to PATH.

        Raises:
            FFmpegNotFoundError: If no ffmpeg executable can be located.
        """
        exe = ffmpeg or shutil.which("ffmpeg")
        if exe is None:
            raise FFmpegNotFoundError(
                "ffmpeg not found on PATH; install FFmpeg or pass ffmpeg=... "
                "(video export only — interactive replay does not need it)."
            )
        self._width = width
        self._height = height
        self._closed = False
        cmd = [
            exe,
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgba",
            "-video_size",
            f"{width}x{height}",
            "-framerate",
            str(fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            os.fspath(path),
        ]
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write_frame(self, rgba: bytes) -> None:
        """Write one raw RGBA8 frame to the encoder.

        Args:
            rgba: Row-major RGBA8 pixels, length ``width*height*4``.

        Raises:
            ValueError: If the frame size does not match the configured size.
            BrokenPipeError: If the encoder has already exited.
        """
        expected = self._width * self._height * 4
        if len(rgba) != expected:
            raise ValueError(f"frame must be {expected} bytes, got {len(rgba)}")
        assert self._proc.stdin is not None
        self._proc.stdin.write(rgba)

    def close(self) -> None:
        """Flush and close FFmpeg's stdin, then wait for encoding to finish.

        Raises:
            RuntimeError: If FFmpeg exits with a non-zero status.
        """
        if self._closed:
            return
        self._closed = True
        if self._proc.stdin is not None:
            self._proc.stdin.close()
        code = self._proc.wait()
        if code != 0:
            raise RuntimeError(f"ffmpeg exited with status {code}")

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
