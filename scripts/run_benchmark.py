#!/usr/bin/env python3
"""Run a benchmark sweep across backends and boid counts; export plots.

Thin wrapper around :func:`boidforge.cli.benchmark_main`. Equivalent to the
installed ``boidforge-benchmark`` console script.

Example:
    python scripts/run_benchmark.py --backends naive-l1 spatial-hash-l2 native-l3 \
        --boids 1000 5000 20000 --out benchmarks/results
"""

from __future__ import annotations

import sys

from boidforge.cli import benchmark_main

if __name__ == "__main__":
    raise SystemExit(benchmark_main(sys.argv[1:]))
