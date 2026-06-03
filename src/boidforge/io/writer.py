"""Sequential ``.bfs`` writer (compute layer).

:class:`FrameWriter` is the only sanctioned way for a solver to persist state.
It is append-only: write the global header once, then stream frames. The
component-major payload lets each call dump the four state arrays with no
repacking. The writer performs no physics and no rendering.
"""

from __future__ import annotations

import os
import struct
from types import TracebackType
from typing import BinaryIO

from boidforge.core.config import SimulationConfig
from boidforge.core.state import SimulationState
from boidforge.io.format import (
    FRAME_HEADER_STRUCT,
    HEADER_SIZE,
    StreamHeader,
)

# Byte offset of the int32 ``frame_count`` field within the global header.
_FRAME_COUNT_OFFSET = 24


class FrameWriter:
    """Append-only writer for a single ``.bfs`` stream.

    Intended use as a context manager::

        with FrameWriter(path, config) as w:
            for t in range(config.steps):
                solver.step(state)
                w.write(t, state)

    On close, the global header's ``frame_count`` is back-patched with the
    number of frames written.
    """

    def __init__(self, path: str | os.PathLike[str], config: SimulationConfig) -> None:
        """Open ``path`` for writing and emit the global header.

        Args:
            path: Destination ``.bfs`` file path.
            config: Run configuration; supplies ``dt``, ``seed``, and the
                ``max_boids`` bound recorded in the header.
        """
        self._header = StreamHeader(
            max_boids=config.n_boids,
            dt=float(config.dt),
            seed=int(config.seed),
            frame_count=-1,
        )
        self._frames = 0
        self._closed = False
        self._fh: BinaryIO = open(path, "wb")
        self._fh.write(self._header.pack())

    def write(self, timestep: int, state: SimulationState) -> None:
        """Append one frame record for ``state`` at ``timestep``.

        Writes the frame header followed by ``px``, ``py``, ``vx``, ``vy`` in
        component-major order. Buffers must be C-contiguous ``float32``.

        Args:
            timestep: Monotonic timestep index for this frame.
            state: Current simulation state to snapshot.
        """
        self._fh.write(struct.pack(FRAME_HEADER_STRUCT, int(timestep), state.n))
        # Component-major (SoA): each array is already C-contiguous float32,
        # so tobytes() is a straight memory dump with no repacking.
        self._fh.write(state.px.tobytes())
        self._fh.write(state.py.tobytes())
        self._fh.write(state.vx.tobytes())
        self._fh.write(state.vy.tobytes())
        self._frames += 1

    @property
    def frames_written(self) -> int:
        """Number of frames appended so far.

        Returns:
            The running frame count.
        """
        return self._frames

    def close(self) -> None:
        """Back-patch ``frame_count`` into the header and close the file."""
        if self._closed:
            return
        self._fh.flush()
        self._fh.seek(_FRAME_COUNT_OFFSET)
        self._fh.write(struct.pack("<i", self._frames))
        self._header.frame_count = self._frames
        self._fh.close()
        self._closed = True

    def header(self) -> StreamHeader:
        """Return the header currently associated with this stream.

        Returns:
            The :class:`StreamHeader` written at open time (``frame_count`` is
            updated after :meth:`close`).
        """
        return self._header

    def __enter__(self) -> FrameWriter:
        """Enter the context manager.

        Returns:
            This writer.
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


# Re-export so callers can compute frame strides without importing format.
__all__ = ["FrameWriter", "HEADER_SIZE"]
