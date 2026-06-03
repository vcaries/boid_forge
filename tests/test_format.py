"""Tests for the ``.bfs`` format constants and header serialization."""

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


def test_header_pack_parse_roundtrip() -> None:
    """A packed header parses back equal; ``dt`` is stored as float32."""
    header = fmt.StreamHeader(dt=0.05, seed=7, max_boids=1000, frame_count=42)
    raw = header.pack()
    assert len(raw) == fmt.HEADER_SIZE

    parsed = fmt.StreamHeader.parse(raw)
    # Integer/byte fields round-trip exactly.
    assert (parsed.version, parsed.flags, parsed.dim, parsed.dtype_code) == (
        header.version,
        header.flags,
        header.dim,
        header.dtype_code,
    )
    assert parsed.max_boids == header.max_boids
    assert parsed.seed == header.seed
    assert parsed.frame_count == header.frame_count
    # dt is intentionally a 32-bit field (matches the solver's float32 math).
    assert parsed.dt == pytest.approx(header.dt, rel=1e-6)


def test_header_dt_exact_when_float32_representable() -> None:
    """A float32-exact dt round-trips with full equality."""
    header = fmt.StreamHeader(dt=0.0625, seed=1, max_boids=10, frame_count=3)
    assert fmt.StreamHeader.parse(header.pack()) == header


def test_parse_rejects_bad_magic() -> None:
    """A buffer without the BFS1 magic is rejected."""
    raw = bytearray(fmt.StreamHeader().pack())
    raw[0:4] = b"XXXX"
    with pytest.raises(ValueError, match="magic"):
        fmt.StreamHeader.parse(bytes(raw))


def test_parse_rejects_wrong_size() -> None:
    """A buffer of the wrong length is rejected before unpacking."""
    with pytest.raises(ValueError, match="32 bytes"):
        fmt.StreamHeader.parse(b"\x00" * 16)
