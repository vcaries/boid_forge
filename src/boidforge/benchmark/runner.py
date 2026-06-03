"""Benchmark harness: measure per-frame cost and scaling per backend.

The runner advances each backend for a fixed number of timesteps (without
writing to disk, to isolate compute cost), records per-frame wall time, and
derives FPS-equivalent and cross-backend speedups. Results serialize to
CSV/JSON for reproducible reporting.
"""

from __future__ import annotations

import csv
import json
import os
import statistics
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field

from boidforge.core.config import SimulationConfig
from boidforge.solver import SOLVERS


@dataclass(slots=True)
class BenchmarkResult:
    """Timing result for one ``(backend, n_boids)`` measurement.

    Attributes:
        backend: Solver name (e.g. ``"native-l4"``).
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

        Builds a config for ``n_boids``, seeds the state once, advances
        ``warmup`` untimed steps, then times each of ``steps`` steps
        individually with a monotonic clock. No frames are written to disk so
        that only compute cost is captured.

        Args:
            backend: Solver name to instantiate and time.
            n_boids: Number of boids for this measurement.

        Returns:
            The :class:`BenchmarkResult` for this configuration.

        Raises:
            KeyError: If ``backend`` is not a registered solver name.
            NotImplementedError: If the backend's seeding or step is a stub.
        """
        solver_cls = SOLVERS[backend]
        cfg = SimulationConfig(n_boids=n_boids, steps=self.steps, seed=self.seed)
        solver = solver_cls(cfg)
        state = solver.initialize()

        for _ in range(self.warmup):
            solver.step(state)

        per_frame_ms: list[float] = []
        for _ in range(self.steps):
            t0 = time.perf_counter()
            solver.step(state)
            per_frame_ms.append((time.perf_counter() - t0) * 1000.0)

        mean = statistics.fmean(per_frame_ms)
        std = statistics.pstdev(per_frame_ms) if len(per_frame_ms) > 1 else 0.0
        fps = 1000.0 / mean if mean > 0.0 else float("inf")
        return BenchmarkResult(
            backend=backend,
            n_boids=n_boids,
            steps=self.steps,
            ms_per_frame=mean,
            ms_std=std,
            fps=fps,
        )

    def run(self) -> list[BenchmarkResult]:
        """Execute the full ``backends × boid_counts`` sweep.

        Backends whose seeding or step is not yet implemented are skipped (a
        notice is written to stderr) so the harness runs against any subset of
        completed levels.

        Returns:
            All collected :class:`BenchmarkResult` objects (also stored on
            :attr:`results`).
        """
        self.results = []
        for backend in self.backends:
            for n in self.boid_counts:
                try:
                    result = self.time_backend(backend, n)
                except NotImplementedError:
                    print(
                        f"[skip] {backend} not implemented; skipping N={n}",
                        file=sys.stderr,
                    )
                    break
                self.results.append(result)
        return self.results

    def speedup(self, baseline: str, target: str) -> dict[int, float]:
        """Compute per-``N`` speedup of ``target`` over ``baseline``.

        Args:
            baseline: Reference backend name (e.g. ``"naive-l1"``).
            target: Faster backend name (e.g. ``"native-l4"``).

        Returns:
            Mapping ``n_boids -> (baseline_ms / target_ms)`` for every ``N``
            measured for both backends.
        """
        base_ms = {r.n_boids: r.ms_per_frame for r in self.results if r.backend == baseline}
        targ_ms = {r.n_boids: r.ms_per_frame for r in self.results if r.backend == target}
        ratios: dict[int, float] = {}
        for n in sorted(set(base_ms) & set(targ_ms)):
            if targ_ms[n] > 0.0:
                ratios[n] = base_ms[n] / targ_ms[n]
        return ratios

    def export_csv(self, path: str | os.PathLike[str]) -> None:
        """Write results to a CSV file.

        Args:
            path: Destination CSV path.
        """
        fields = ["backend", "n_boids", "steps", "ms_per_frame", "ms_std", "fps"]
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            for r in self.results:
                writer.writerow(asdict(r))

    def export_json(self, path: str | os.PathLike[str]) -> None:
        """Write results plus run metadata to a JSON file.

        Args:
            path: Destination JSON path.
        """
        payload = {
            "metadata": {
                "backends": list(self.backends),
                "boid_counts": list(self.boid_counts),
                "steps": self.steps,
                "warmup": self.warmup,
                "seed": self.seed,
            },
            "results": [asdict(r) for r in self.results],
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
