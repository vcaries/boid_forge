"""Perceptual colormaps as GPU lookup tables, with no matplotlib dependency.

The renderer maps a scalar per boid (speed or heading) to colour by sampling a
256-entry RGB lookup table uploaded as a 1-D texture. Keeping the tables here —
built from compact anchor gradients and the Turbo polynomial — avoids pulling
matplotlib into the visualization runtime (matplotlib stays a ``benchmark``
concern only).

All tables are ``float32`` arrays of shape ``(256, 3)`` with values in ``[0, 1]``.
"""

from __future__ import annotations

import numpy as np

from boidforge.core.types import FloatArray

#: Number of entries in every generated lookup table.
LUT_SIZE: int = 256

# Anchor gradients: (position in [0,1], (r, g, b) in [0,1]). Interpolated
# linearly to fill the LUT. These approximations are tuned for a dark
# background and a bright, modern aesthetic.
_ANCHORS: dict[str, tuple[tuple[float, tuple[float, float, float]], ...]] = {
    "viridis": (
        (0.0, (0.267, 0.005, 0.329)),
        (0.25, (0.229, 0.322, 0.545)),
        (0.5, (0.127, 0.567, 0.551)),
        (0.75, (0.369, 0.789, 0.383)),
        (1.0, (0.993, 0.906, 0.144)),
    ),
    "inferno": (
        (0.0, (0.001, 0.000, 0.014)),
        (0.25, (0.258, 0.039, 0.406)),
        (0.5, (0.578, 0.148, 0.404)),
        (0.75, (0.865, 0.317, 0.226)),
        (0.9, (0.988, 0.645, 0.040)),
        (1.0, (0.988, 0.998, 0.645)),
    ),
    "magma": (
        (0.0, (0.001, 0.000, 0.014)),
        (0.25, (0.232, 0.059, 0.437)),
        (0.5, (0.550, 0.161, 0.506)),
        (0.75, (0.872, 0.288, 0.408)),
        (0.9, (0.996, 0.624, 0.427)),
        (1.0, (0.987, 0.991, 0.749)),
    ),
    # A cool electric-blue to magenta "neon" ramp.
    "neon": (
        (0.0, (0.02, 0.05, 0.20)),
        (0.35, (0.0, 0.55, 0.95)),
        (0.6, (0.25, 0.95, 0.90)),
        (0.8, (0.85, 0.30, 0.95)),
        (1.0, (1.0, 0.85, 0.95)),
    ),
    # Deep teal to warm white "aurora".
    "aurora": (
        (0.0, (0.02, 0.10, 0.12)),
        (0.4, (0.05, 0.65, 0.45)),
        (0.7, (0.45, 0.95, 0.55)),
        (0.9, (0.85, 0.98, 0.70)),
        (1.0, (0.98, 1.0, 0.95)),
    ),
    # Blackbody-ish "fire".
    "fire": (
        (0.0, (0.02, 0.0, 0.0)),
        (0.35, (0.6, 0.05, 0.0)),
        (0.6, (0.95, 0.4, 0.05)),
        (0.85, (1.0, 0.85, 0.25)),
        (1.0, (1.0, 1.0, 0.92)),
    ),
    # Single-hue cyan "ice".
    "ice": (
        (0.0, (0.01, 0.03, 0.08)),
        (0.5, (0.10, 0.45, 0.75)),
        (0.8, (0.55, 0.85, 0.98)),
        (1.0, (0.95, 0.99, 1.0)),
    ),
}


def _turbo() -> FloatArray:
    """Build the Turbo colormap from its polynomial approximation.

    Uses Anton Mikhailov's published cubic/quartic fit so the table is exact to
    the Google Turbo reference without shipping a 256x3 data block.

    Returns:
        The Turbo lookup table, shape ``(256, 3)`` ``float32``.
    """
    t = np.linspace(0.0, 1.0, LUT_SIZE, dtype=np.float64)
    r = (
        0.13572138
        + 4.61539260 * t
        - 42.66032258 * t**2
        + 132.13108234 * t**3
        - 152.94239396 * t**4
        + 59.28637943 * t**5
    )
    g = (
        0.09140261
        + 2.19418839 * t
        + 4.84296658 * t**2
        - 14.18503333 * t**3
        + 4.27729857 * t**4
        + 2.82956604 * t**5
    )
    b = (
        0.10667330
        + 12.64194608 * t
        - 60.58204836 * t**2
        + 110.36276771 * t**3
        - 89.90310912 * t**4
        + 27.34824973 * t**5
    )
    lut = np.stack([r, g, b], axis=1)
    return np.clip(lut, 0.0, 1.0).astype(np.float32)


def _from_anchors(anchors: tuple[tuple[float, tuple[float, float, float]], ...]) -> FloatArray:
    """Interpolate an anchor gradient into a full lookup table.

    Args:
        anchors: Ordered ``(position, rgb)`` control points spanning ``[0, 1]``.

    Returns:
        The interpolated lookup table, shape ``(256, 3)`` ``float32``.
    """
    positions = np.array([a[0] for a in anchors], dtype=np.float64)
    colors = np.array([a[1] for a in anchors], dtype=np.float64)
    t = np.linspace(0.0, 1.0, LUT_SIZE, dtype=np.float64)
    channels = [np.interp(t, positions, colors[:, c]) for c in range(3)]
    lut = np.stack(channels, axis=1)
    return np.clip(lut, 0.0, 1.0).astype(np.float32)


def available() -> tuple[str, ...]:
    """List the colormap names this module can build.

    Returns:
        Sorted tuple of valid names for :func:`lookup_table`.
    """
    return ("turbo", *sorted(_ANCHORS))


def lookup_table(name: str) -> FloatArray:
    """Return the 256-entry RGB lookup table for ``name``.

    Args:
        name: A colormap name from :func:`available`.

    Returns:
        Lookup table of shape ``(256, 3)`` ``float32`` in ``[0, 1]``.

    Raises:
        KeyError: If ``name`` is not a known colormap.
    """
    if name == "turbo":
        return _turbo()
    if name not in _ANCHORS:
        raise KeyError(f"unknown colormap {name!r}; choose from {available()}")
    return _from_anchors(_ANCHORS[name])
