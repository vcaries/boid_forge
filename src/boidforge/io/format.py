"""``.bfs`` binary format definition (constants + header descriptor).

The numeric constants and struct layouts below *are* the format specification in
code form; they mirror ``docs/architecture.md`` §"Binary format". They are
concrete because they are declarations, not logic. Header (de)serialization is
defined as an interface here and implemented by the writer/reader.

Layout summary (little-endian, ``float32`` payloads, component-major / SoA)::

    [ 32-byte global header ]
    repeated frame records:
        int32 timestep
        int32 n_boids
        float32 px[N]
        float32 py[N]
        float32 vx[N]
        float32 vy[N]
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

from boidforge.core.types import DIM

#: 4-byte file magic identifying a BoidForge stream, version 1.
MAGIC: bytes = b"BFS1"

#: On-disk format version. Bump on any layout change.
FORMAT_VERSION: int = 1

#: ``flags`` bit: payload is little-endian.
FLAG_LITTLE_ENDIAN: int = 1 << 0

#: ``dtype_code`` value for IEEE-754 single precision.
DTYPE_CODE_FLOAT32: int = 0

#: Number of SoA components per frame (px, py, vx, vy).
N_COMPONENTS: int = 4

#: ``struct`` format for the 32-byte global header (see module docstring).
#: Fields: magic, version, flags, dim, dtype_code, reserved[2], max_boids,
#: dt, seed, frame_count, reserved2[4].
HEADER_STRUCT: str = "<4sHHBB2sifIi4s"

#: Size in bytes of the global header.
HEADER_SIZE: int = struct.calcsize(HEADER_STRUCT)

#: ``struct`` format for a per-frame header: int32 timestep, int32 n_boids.
FRAME_HEADER_STRUCT: str = "<ii"

#: Size in bytes of a per-frame header.
FRAME_HEADER_SIZE: int = struct.calcsize(FRAME_HEADER_STRUCT)

#: Bytes of payload per boid per frame: N_COMPONENTS * sizeof(float32).
BYTES_PER_BOID: int = N_COMPONENTS * 4


def frame_size(n_boids: int) -> int:
    """Total on-disk size of one frame record.

    Args:
        n_boids: Number of boids ``N`` in the frame.

    Returns:
        ``FRAME_HEADER_SIZE + BYTES_PER_BOID * n_boids`` bytes.
    """
    return FRAME_HEADER_SIZE + BYTES_PER_BOID * n_boids


@dataclass(slots=True)
class StreamHeader:
    """In-memory view of the 32-byte global header.

    Attributes:
        version: Format version; expected to equal :data:`FORMAT_VERSION`.
        flags: Bitfield (see :data:`FLAG_LITTLE_ENDIAN`).
        dim: Spatial dimensions (``2``).
        dtype_code: Payload dtype code (see :data:`DTYPE_CODE_FLOAT32`).
        max_boids: Upper bound on boids per frame, or ``0`` if unbounded.
        dt: Integration timestep recorded by the solver.
        seed: RNG seed used for the run (provenance).
        frame_count: Number of frames, or ``-1`` while still streaming.
    """

    version: int = FORMAT_VERSION
    flags: int = FLAG_LITTLE_ENDIAN
    dim: int = DIM
    dtype_code: int = DTYPE_CODE_FLOAT32
    max_boids: int = 0
    dt: float = 0.0
    seed: int = 0
    frame_count: int = -1

    def pack(self) -> bytes:
        """Serialize this header to its 32-byte on-disk form.

        Returns:
            Exactly :data:`HEADER_SIZE` bytes.
        """
        return struct.pack(
            HEADER_STRUCT,
            MAGIC,
            self.version,
            self.flags,
            self.dim,
            self.dtype_code,
            b"\x00\x00",
            self.max_boids,
            self.dt,
            self.seed,
            self.frame_count,
            b"\x00\x00\x00\x00",
        )

    @classmethod
    def parse(cls, raw: bytes) -> StreamHeader:
        """Deserialize and validate a 32-byte global header.

        Args:
            raw: Exactly :data:`HEADER_SIZE` bytes read from the stream start.

        Returns:
            The parsed :class:`StreamHeader`.

        Raises:
            ValueError: If the size, magic, version, or dtype is unsupported.
        """
        if len(raw) != HEADER_SIZE:
            raise ValueError(f"header must be {HEADER_SIZE} bytes, got {len(raw)}")
        (
            magic,
            version,
            flags,
            dim,
            dtype_code,
            _reserved,
            max_boids,
            dt,
            seed,
            frame_count,
            _reserved2,
        ) = struct.unpack(HEADER_STRUCT, raw)
        if magic != MAGIC:
            raise ValueError(f"not a BoidForge stream (bad magic {magic!r})")
        if version != FORMAT_VERSION:
            raise ValueError(f"unsupported format version {version}")
        if dtype_code != DTYPE_CODE_FLOAT32:
            raise ValueError(f"unsupported dtype code {dtype_code}")
        return cls(
            version=version,
            flags=flags,
            dim=dim,
            dtype_code=dtype_code,
            max_boids=max_boids,
            dt=dt,
            seed=seed,
            frame_count=frame_count,
        )
