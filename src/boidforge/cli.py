"""Console entry points for the three top-level workflows.

These are thin argument-parsing front ends that delegate to the subsystems:

* :func:`simulate_main` → :mod:`boidforge.solver` (compute, writes ``.bfs``)
* :func:`benchmark_main` → :mod:`boidforge.benchmark`
* :func:`replay_main` → :mod:`boidforge.viz` (reads ``.bfs``, renders)

Registered in ``pyproject.toml`` as ``boidforge-simulate``,
``boidforge-benchmark``, and ``boidforge-replay``.
"""

from __future__ import annotations

from collections.abc import Sequence


def simulate_main(argv: Sequence[str] | None = None) -> int:
    """Run a simulation and write a ``.bfs`` stream.

    Args:
        argv: Command-line arguments (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code.
    """
    raise NotImplementedError


def benchmark_main(argv: Sequence[str] | None = None) -> int:
    """Run a benchmark sweep and export results + plots.

    Args:
        argv: Command-line arguments (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code.
    """
    raise NotImplementedError


def replay_main(argv: Sequence[str] | None = None) -> int:
    """Replay a ``.bfs`` stream to a window or a video file.

    Args:
        argv: Command-line arguments (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code.
    """
    raise NotImplementedError
