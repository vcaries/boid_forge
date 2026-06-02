"""Binary ``.bfs`` stream I/O — the sole contract between solver and visualizer.

This subpackage defines the format (:mod:`boidforge.io.format`), the sequential
:class:`~boidforge.io.writer.FrameWriter` used by the compute layer, and the
sequential :class:`~boidforge.io.reader.FrameReader` used by the post-processing
layer. It depends only on :mod:`boidforge.core` and NumPy so that neither
subsystem couples to the other through it.

See ``docs/architecture.md`` §"Binary format" for the byte layout and rationale.
"""

from __future__ import annotations

from boidforge.io.format import (
    FORMAT_VERSION,
    FRAME_HEADER_SIZE,
    HEADER_SIZE,
    MAGIC,
    StreamHeader,
)
from boidforge.io.reader import FrameReader
from boidforge.io.writer import FrameWriter

__all__ = [
    "FORMAT_VERSION",
    "MAGIC",
    "HEADER_SIZE",
    "FRAME_HEADER_SIZE",
    "StreamHeader",
    "FrameWriter",
    "FrameReader",
]
