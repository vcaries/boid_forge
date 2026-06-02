"""Sequential ``.bfs`` reader (post-processing layer).

:class:`FrameReader` is the only sanctioned way for the visualizer to obtain
boid state. It reads frames in order and returns zero-copy ``float32`` views
where possible. It performs no physics: a frame that is not on disk is not
computed.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass
from types import TracebackType

from boidforge.core.state import SimulationState
from boidforge.io.format import StreamHeader


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
        raise NotImplementedError

    @property
    def header(self) -> StreamHeader:
        """Parsed global header for this stream.

        Returns:
            The :class:`StreamHeader` read at open time.
        """
        raise NotImplementedError

    def read_frame(self) -> Frame | None:
        """Read the next frame in sequence.

        Returns:
            The next :class:`Frame`, or ``None`` at end of stream.
        """
        raise NotImplementedError

    def seek_frame(self, index: int) -> None:
        """Position the stream at frame ``index``.

        Uses the sidecar ``.bfx`` index when present; otherwise walks frame
        headers from the current position. Random seeking is a visualizer
        convenience and never required for sequential replay.

        Args:
            index: Zero-based frame index to seek to.
        """
        raise NotImplementedError

    def __iter__(self) -> Iterator[Frame]:
        """Iterate frames from the current position to end of stream.

        Yields:
            Each decoded :class:`Frame` in order.
        """
        raise NotImplementedError

    def close(self) -> None:
        """Close the underlying file."""
        raise NotImplementedError

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
