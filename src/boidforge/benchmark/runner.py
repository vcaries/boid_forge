"""Benchmark harness: measure per-frame cost and scaling per backend.

The runner advances each backend for a fixed number of timesteps (without
writing to disk, to isolate compute cost), records per-frame wall time, and
derives FPS-equivalent and cross-backend speedups. Results serialize to
CSV/JSON for reproducible reporting.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass, field


@dataclass(slots=True)
class BenchmarkResult:
    """Timing result for one ``(backend, n_boids)`` measurement.

    Attributes:
        backend: Solver name (e.g. ``"native-l3"``).
        n_boids: Number of boids simulated.
        steps: Number of timesteps measured.
        ms_per_frame: Mean per-frame wall time in milliseconds.
        ms_std: Standard deviation of per-frame time in milliseconds.
        fps: FPS-equivalent, ``1000 / ms_per_frame``.
    """

    backend: str
    n_boids: int
    steps: int
    ms_per_frame: float
    ms_std: float
    fps: float


@dataclass(slots=True)
class BenchmarkRunner:
    """Runs timing sweeps across backends and boid counts.

    Attributes:
        backends: Backend names to measure (keys of ``solver.SOLVERS``).
        boid_counts: Values of ``N`` to sweep.
        steps: Timesteps to advance per measurement.
        warmup: Untimed warmup steps before measurement begins.
        seed: RNG seed shared by every measured run.
        results: Collected results, populated by :meth:`run`.
    """

    backends: Sequence[str]
    boid_counts: Sequence[int]
    steps: int = 200
    warmup: int = 10
    seed: int = 0
    results: list[BenchmarkResult] = field(default_factory=list)

    def time_backend(self, backend: str, n_boids: int) -> BenchmarkResult:
        """Measure one backend at one boid count.

        Args:
            backend: Solver name to instantiate and time.
            n_boids: Number of boids for this measurement.

        Returns:
            The :class:`BenchmarkResult` for this configuration.
        """
        raise NotImplementedError

    def run(self) -> list[BenchmarkResult]:
        """Execute the full ``backends × boid_counts`` sweep.

        Returns:
            All collected :class:`BenchmarkResult` objects (also stored on
            :attr:`results`).
        """
        raise NotImplementedError

    def speedup(self, baseline: str, target: str) -> dict[int, float]:
        """Compute per-``N`` speedup of ``target`` over ``baseline``.

        Args:
            baseline: Reference backend name (e.g. ``"naive-l1"``).
            target: Faster backend name (e.g. ``"native-l3"``).

        Returns:
            Mapping ``n_boids -> (baseline_ms / target_ms)``.
        """
        raise NotImplementedError

    def export_csv(self, path: str | os.PathLike[str]) -> None:
        """Write results to a CSV file.

        Args:
            path: Destination CSV path.
        """
        raise NotImplementedError

    def export_json(self, path: str | os.PathLike[str]) -> None:
        """Write results plus run metadata to a JSON file.

        Args:
            path: Destination JSON path.
        """
        raise NotImplementedError
