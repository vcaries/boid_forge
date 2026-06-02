"""Matplotlib plotting for benchmark results.

Pure presentation over :class:`~boidforge.benchmark.runner.BenchmarkResult`
data. Figures are written to disk (PNG/SVG); nothing is committed (see
``.gitignore``). This module imports matplotlib lazily so that importing the
benchmark package does not require it.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

from boidforge.benchmark.runner import BenchmarkResult


def plot_scaling(
    results: Sequence[BenchmarkResult],
    path: str | os.PathLike[str],
) -> None:
    """Plot per-frame time (ms) vs number of boids, per backend.

    Uses log-log axes to make the O(N²) vs ~O(N) regimes visually distinct.

    Args:
        results: Measurements to plot, grouped internally by backend.
        path: Output image path.
    """
    raise NotImplementedError


def plot_speedup(
    speedups: dict[str, dict[int, float]],
    path: str | os.PathLike[str],
) -> None:
    """Plot speedup ratios across backends as grouped bars.

    Args:
        speedups: Mapping ``"target_vs_baseline" -> {n_boids: ratio}``.
        path: Output image path.
    """
    raise NotImplementedError
