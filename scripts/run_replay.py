#!/usr/bin/env python3
"""Replay a ``.bfs`` stream to a window or export it to video.

Thin wrapper around :func:`boidforge.cli.replay_main`. Equivalent to the
installed ``boidforge-replay`` console script.

Example:
    python scripts/run_replay.py runs/flock.bfs --export out/flock.mp4 --fps 60
"""

from __future__ import annotations

import sys

from boidforge.cli import replay_main

if __name__ == "__main__":
    raise SystemExit(replay_main(sys.argv[1:]))
