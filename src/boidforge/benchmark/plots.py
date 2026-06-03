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
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_backend: dict[str, list[BenchmarkResult]] = {}
    for r in results:
        by_backend.setdefault(r.backend, []).append(r)

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    for backend in sorted(by_backend):
        series = sorted(by_backend[backend], key=lambda r: r.n_boids)
        xs = [r.n_boids for r in series]
        ys = [r.ms_per_frame for r in series]
        ax.plot(xs, ys, marker="o", label=backend)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of boids (N)")
    ax.set_ylabel("Per-frame time (ms)")
    ax.set_title("Solver scaling: per-frame time vs N")
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_speedup(
    speedups: dict[str, dict[int, float]],
    path: str | os.PathLike[str],
) -> None:
    """Plot speedup ratios across backends as grouped bars.

    Args:
        speedups: Mapping ``"target_vs_baseline" -> {n_boids: ratio}``.
        path: Output image path.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    all_n = sorted({n for ratios in speedups.values() for n in ratios})
    series = sorted(speedups)
    n_series = max(len(series), 1)
    group_width = 0.8
    bar_width = group_width / n_series

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    for idx, label in enumerate(series):
        ratios = speedups[label]
        xs = [i + (idx - (n_series - 1) / 2.0) * bar_width for i in range(len(all_n))]
        ys = [ratios.get(n, 0.0) for n in all_n]
        ax.bar(xs, ys, width=bar_width, label=label)

    ax.axhline(1.0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_xticks(range(len(all_n)))
    ax.set_xticklabels([str(n) for n in all_n])
    ax.set_xlabel("Number of boids (N)")
    ax.set_ylabel("Speedup (×, baseline / target)")
    ax.set_title("Backend speedup vs baseline")
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
