#!/usr/bin/env python3
"""Run a Boids simulation and write a ``.bfs`` stream.

Thin wrapper around :func:`boidforge.cli.simulate_main`. Equivalent to the
installed ``boidforge-simulate`` console script.

Example:
    python scripts/run_simulation.py --backend native-l4 --boids 50000 \
        --steps 1200 --out runs/flock.bfs
"""

from __future__ import annotations

import sys

from boidforge.cli import simulate_main

if __name__ == "__main__":
    raise SystemExit(simulate_main(sys.argv[1:]))
