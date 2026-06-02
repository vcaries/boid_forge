"""Benchmarking: time backends across boid counts and plot scaling/speedup.

:mod:`boidforge.benchmark.runner` measures per-frame time and FPS-equivalent for
each solver across a sweep of ``N``; :mod:`boidforge.benchmark.plots` renders the
matplotlib figures. This subpackage imports the solver layer but never the
visualization layer.
"""

from __future__ import annotations

from boidforge.benchmark.runner import BenchmarkResult, BenchmarkRunner

__all__ = ["BenchmarkResult", "BenchmarkRunner"]
