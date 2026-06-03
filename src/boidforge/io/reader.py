"""Sequential ``.bfs`` reader (post-processing layer).

:class:`FrameReader` is the only sanctioned way for the visualizer to obtain
boid state. It reads frames in order and returns zero-copy ``float32`` views
where possible. It performs no physics: a frame that is not on disk is not
computed.
"""

from __future__ import annotations

import os
import struct
from collections.abc import Iterator
from dataclasses import dataclass
from types import TracebackType
from typing import BinaryIO

import numpy as np

from boidforge.core.state import SimulationState
from boidforge.core.types import DTYPE, FloatArray
from boidforge.io.format import (
    FRAME_HEADER_SIZE,
    FRAME_HEADER_STRUCT,
    HEADER_SIZE,
    StreamHeader,
    frame_size,
)


@dataclass(slots=True)
class Frame:
    """A single decoded frame.

    Attributes:
        timestep: Timestep index recorded by the solver.
        state: SoA state for this frame (views into the read buffer).
    """

    timestep: int
    state: SimulationState


class FrameReader:
    """Sequential reader over a ``.bfs`` stream.

    Intended use as a context manager / iterator::

        with FrameReader(path) as r:
            for frame in r:
                renderer.draw(frame)
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        """Open ``path`` and parse the global header.

        Args:
            path: Source ``.bfs`` file path.

        Raises:
            ValueError: If the header magic/version/dtype is unsupported.
        """
        self._fh: BinaryIO = open(path, "rb")
        self._header = StreamHeader.parse(self._fh.read(HEADER_SIZE))
        self._closed = False

    @property
    def header(self) -> StreamHeader:
        """Parsed global header for this stream.

        Returns:
            The :class:`StreamHeader` read at open time.
        """
        return self._header

    def _read_component(self, n: int) -> FloatArray:
        """Read one ``float32[n]`` SoA component from the current position."""
        nbytes = n * DTYPE.itemsize
        buf = self._fh.read(nbytes)
        if len(buf) < nbytes:
            raise ValueError("truncated frame payload")
        return np.frombuffer(buf, dtype=DTYPE)

    def read_frame(self) -> Frame | None:
        """Read the next frame in sequence.

        Returns:
            The next :class:`Frame`, or ``None`` at end of stream.
        """
        hdr = self._fh.read(FRAME_HEADER_SIZE)
        if not hdr:
            return None
        if len(hdr) < FRAME_HEADER_SIZE:
            raise ValueError("truncated frame header")
        timestep, n = struct.unpack(FRAME_HEADER_STRUCT, hdr)
        state = SimulationState(
            px=self._read_component(n),
            py=self._read_component(n),
            vx=self._read_component(n),
            vy=self._read_component(n),
        )
        return Frame(timestep=timestep, state=state)

    def seek_frame(self, index: int) -> None:
        """Position the stream at frame ``index``.

        Uses fixed-stride arithmetic when the header records a ``max_boids``
        bound (every frame is the same size, as produced by this package's
        writer). Variable-length streams would require a ``.bfx`` index.

        Args:
            index: Zero-based frame index to seek to.

        Raises:
            NotImplementedError: If the stream has no fixed frame stride.
        """
        if self._header.max_boids <= 0:
            raise NotImplementedError("seek requires a fixed max_boids or a .bfx index")
        stride = frame_size(self._header.max_boids)
        self._fh.seek(HEADER_SIZE + index * stride)

    def __iter__(self) -> Iterator[Frame]:
        """Iterate frames from the current position to end of stream.

        Yields:
            Each decoded :class:`Frame` in order.
        """
        while True:
            frame = self.read_frame()
            if frame is None:
                return
            yield frame

    def close(self) -> None:
        """Close the underlying file."""
        if not self._closed:
            self._fh.close()
            self._closed = True

    def __enter__(self) -> FrameReader:
        """Enter the context manager.

        Returns:
            This reader.
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Close the stream on context exit."""
        self.close()
