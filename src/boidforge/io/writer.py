"""Sequential ``.bfs`` writer (compute layer).

:class:`FrameWriter` is the only sanctioned way for a solver to persist state.
It is append-only: write the global header once, then stream frames. The
component-major payload lets each call dump the four state arrays with no
repacking. The writer performs no physics and no rendering.
"""

from __future__ import annotations

import os
from types import TracebackType

from boidforge.core.config import SimulationConfig
from boidforge.core.state import SimulationState
from boidforge.io.format import StreamHeader


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
        raise NotImplementedError

    def write(self, timestep: int, state: SimulationState) -> None:
        """Append one frame record for ``state`` at ``timestep``.

        Writes the frame header followed by ``px``, ``py``, ``vx``, ``vy`` in
        component-major order. Buffers must be C-contiguous ``float32``.

        Args:
            timestep: Monotonic timestep index for this frame.
            state: Current simulation state to snapshot.
        """
        raise NotImplementedError

    @property
    def frames_written(self) -> int:
        """Number of frames appended so far.

        Returns:
            The running frame count.
        """
        raise NotImplementedError

    def close(self) -> None:
        """Back-patch ``frame_count`` into the header and close the file."""
        raise NotImplementedError

    def header(self) -> StreamHeader:
        """Return the header currently associated with this stream.

        Returns:
            The :class:`StreamHeader` written at open time.
        """
        raise NotImplementedError

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
