"""Tests for the ``.bfs`` format constants and (pending) header serialization."""

from __future__ import annotations

import pytest

from boidforge.io import format as fmt


def test_header_size_is_32_bytes() -> None:
    """The global header is exactly 32 bytes."""
    assert fmt.HEADER_SIZE == 32


def test_frame_header_size_is_8_bytes() -> None:
    """A per-frame header is two int32 = 8 bytes."""
    assert fmt.FRAME_HEADER_SIZE == 8


def test_bytes_per_boid() -> None:
    """Each boid contributes 4 float32 components = 16 bytes per frame."""
    assert fmt.BYTES_PER_BOID == 16


def test_frame_size_formula() -> None:
    """frame_size = header + 16 * N."""
    assert fmt.frame_size(0) == fmt.FRAME_HEADER_SIZE
    assert fmt.frame_size(1000) == fmt.FRAME_HEADER_SIZE + 16 * 1000


def test_magic_and_version() -> None:
    """Magic and version match the v1 specification."""
    assert fmt.MAGIC == b"BFS1"
    assert fmt.FORMAT_VERSION == 1


@pytest.mark.xfail(raises=NotImplementedError, strict=True, reason="skeleton")
def test_header_pack_parse_roundtrip() -> None:
    """A packed header parses back to an equal header (once implemented)."""
    header = fmt.StreamHeader(dt=0.05, seed=7, max_boids=1000, frame_count=42)
    raw = header.pack()
    assert len(raw) == fmt.HEADER_SIZE
    assert fmt.StreamHeader.parse(raw) == header
