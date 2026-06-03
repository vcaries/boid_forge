"""Console entry points for the three top-level workflows.

These are thin argument-parsing front ends that delegate to the subsystems:

* :func:`simulate_main` → :mod:`boidforge.solver` (compute, writes ``.bfs``)
* :func:`benchmark_main` → :mod:`boidforge.benchmark`
* :func:`replay_main` → :mod:`boidforge.viz` (reads ``.bfs``, renders)

Registered in ``pyproject.toml`` as ``boidforge-simulate``,
``boidforge-benchmark``, and ``boidforge-replay``.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from boidforge.core.config import BoundaryMode, SimulationConfig
from boidforge.solver import SOLVERS


def simulate_main(argv: Sequence[str] | None = None) -> int:
    """Run a simulation and write a ``.bfs`` stream.

    Args:
        argv: Command-line arguments (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(
        prog="boidforge-simulate",
        description="Advance a boid simulation and write a .bfs stream.",
    )
    parser.add_argument("--backend", default="naive-l1", choices=sorted(SOLVERS))
    parser.add_argument("-n", "--boids", type=int, default=500)
    parser.add_argument("-s", "--steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--boundary", default="wrap", choices=[m.value for m in BoundaryMode])
    parser.add_argument("--world-width", type=float, default=1920.0)
    parser.add_argument("--world-height", type=float, default=1080.0)
    parser.add_argument("-o", "--out", required=True, help="Destination .bfs path.")
    args = parser.parse_args(argv)

    cfg = SimulationConfig(
        n_boids=args.boids,
        steps=args.steps,
        dt=args.dt,
        world_width=args.world_width,
        world_height=args.world_height,
        boundary=BoundaryMode(args.boundary),
        seed=args.seed,
    )
    solver = SOLVERS[args.backend](cfg)
    frames = solver.run(args.out)
    print(f"{args.backend}: wrote {frames} frames (N={cfg.n_boids}, seed={cfg.seed}) -> {args.out}")
    return 0


def benchmark_main(argv: Sequence[str] | None = None) -> int:
    """Run a benchmark sweep and export results + plots.

    Args:
        argv: Command-line arguments (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code.
    """
    from boidforge.benchmark.plots import plot_scaling, plot_speedup
    from boidforge.benchmark.runner import BenchmarkRunner

    parser = argparse.ArgumentParser(
        prog="boidforge-benchmark",
        description="Sweep backends × boid counts and export timings + plots.",
    )
    parser.add_argument("--backends", nargs="+", default=sorted(SOLVERS), choices=sorted(SOLVERS))
    parser.add_argument("-n", "--boids", nargs="+", type=int, default=[100, 200, 500, 1000, 2000])
    parser.add_argument("-s", "--steps", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default=".", help="Directory for CSV/JSON/plots.")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)

    import os

    os.makedirs(args.out_dir, exist_ok=True)

    runner = BenchmarkRunner(
        backends=args.backends,
        boid_counts=args.boids,
        steps=args.steps,
        warmup=args.warmup,
        seed=args.seed,
    )
    results = runner.run()
    if not results:
        print("No backends produced results (all skipped).")
        return 1

    csv_path = os.path.join(args.out_dir, "benchmark.csv")
    json_path = os.path.join(args.out_dir, "benchmark.json")
    runner.export_csv(csv_path)
    runner.export_json(json_path)
    print(f"Wrote {csv_path} and {json_path}")

    for r in results:
        print(
            f"  {r.backend:>16}  N={r.n_boids:<7} {r.ms_per_frame:8.3f} ms/frame  {r.fps:8.1f} fps"
        )

    if not args.no_plots:
        plot_scaling(results, os.path.join(args.out_dir, "scaling.png"))
        measured = [b for b in args.backends if any(r.backend == b for r in results)]
        speedups: dict[str, dict[int, float]] = {}
        if len(measured) > 1:
            baseline = measured[0]
            for target in measured[1:]:
                speedups[f"{target}_vs_{baseline}"] = runner.speedup(baseline, target)
            plot_speedup(speedups, os.path.join(args.out_dir, "speedup.png"))
        print(f"Wrote plots to {args.out_dir}")
    return 0


def replay_main(argv: Sequence[str] | None = None) -> int:
    """Replay a ``.bfs`` stream to a window or a video file.

    Args:
        argv: Command-line arguments (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code.
    """
    from boidforge.viz.colormaps import available
    from boidforge.viz.replay import ReplayConfig, ReplayEngine

    parser = argparse.ArgumentParser(
        prog="boidforge-replay",
        description="Replay a .bfs stream interactively or render it to video.",
    )
    parser.add_argument("stream", help="Source .bfs file to replay.")
    parser.add_argument(
        "-o", "--export", default=None, help="Render offscreen to this video file (needs ffmpeg)."
    )
    parser.add_argument("--fps", type=int, default=60, help="Playback/export frame rate.")
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--colormap", default="turbo", choices=sorted(available()))
    parser.add_argument("--color-mode", default="speed", choices=["speed", "heading", "uniform"])
    parser.add_argument("--crf", type=int, default=18, help="x264 quality (lower = better).")
    parser.add_argument("--max-frames", type=int, default=0, help="Cap frames decoded (0 = all).")
    parser.add_argument("--no-loop", action="store_true", help="Do not loop in interactive mode.")
    parser.add_argument(
        "--no-auto-speed", action="store_true", help="Disable colour speed auto-calibration."
    )
    args = parser.parse_args(argv)

    cfg = ReplayConfig(
        fps=args.fps,
        loop=not args.no_loop,
        export_path=args.export,
        width=args.width,
        height=args.height,
        colormap=args.colormap,
        color_mode=args.color_mode,
        crf=args.crf,
        max_frames=args.max_frames,
        auto_speed=not args.no_auto_speed,
    )
    engine = ReplayEngine(args.stream, cfg)
    engine.run()
    if args.export:
        print(f"Rendered {args.stream} -> {args.export}")
    return 0
