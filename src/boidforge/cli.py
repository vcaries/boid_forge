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

    rules = parser.add_argument_group("boid rules", "Interaction radii, weights, and clamps.")
    rules.add_argument("--r-sep", type=float, default=12.0, help="Separation radius.")
    rules.add_argument("--r-ali", type=float, default=30.0, help="Alignment radius.")
    rules.add_argument("--r-coh", type=float, default=30.0, help="Cohesion radius.")
    rules.add_argument("--w-sep", type=float, default=1.5, help="Separation weight.")
    rules.add_argument("--w-ali", type=float, default=1.0, help="Alignment weight.")
    rules.add_argument("--w-coh", type=float, default=1.0, help="Cohesion weight.")
    rules.add_argument("--max-speed", type=float, default=180.0, help="Upper speed clamp.")
    rules.add_argument("--min-speed", type=float, default=40.0, help="Lower speed clamp.")
    rules.add_argument("--max-force", type=float, default=220.0, help="Per-step steering clamp.")

    parser.add_argument("-o", "--out", required=True, help="Destination .bfs path.")
    args = parser.parse_args(argv)

    cfg = SimulationConfig(
        n_boids=args.boids,
        steps=args.steps,
        dt=args.dt,
        world_width=args.world_width,
        world_height=args.world_height,
        boundary=BoundaryMode(args.boundary),
        r_sep=args.r_sep,
        r_ali=args.r_ali,
        r_coh=args.r_coh,
        w_sep=args.w_sep,
        w_ali=args.w_ali,
        w_coh=args.w_coh,
        max_speed=args.max_speed,
        min_speed=args.min_speed,
        max_force=args.max_force,
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
    parser.add_argument(
        "--color-mode", default="speed", choices=["speed", "heading", "uniform", "density"]
    )
    parser.add_argument("--crf", type=int, default=18, help="x264 quality (lower = better).")
    parser.add_argument(
        "--start-frame", type=int, default=0, help="First frame to replay/export (0 = start)."
    )
    parser.add_argument(
        "--max-frames", type=int, default=0, help="Frames to decode from --start-frame (0 = all)."
    )
    parser.add_argument("--no-loop", action="store_true", help="Do not loop in interactive mode.")
    parser.add_argument(
        "--no-auto-speed", action="store_true", help="Disable colour speed auto-calibration."
    )

    look = parser.add_argument_group(
        "appearance",
        "Live-tunable look (mirrors the interactive panel). Each defaults to the "
        "auto/computed value; pass one to pin it so an export matches what you tuned.",
    )
    look.add_argument("--point-size", type=float, default=None, help="Sprite diameter (px).")
    look.add_argument("--glow", type=float, default=None, help="Sprite halo strength.")
    look.add_argument("--intensity", type=float, default=None, help="Emission multiplier.")
    look.add_argument("--trail-decay", type=float, default=None, help="Trail retention 0..1.")
    look.add_argument(
        "--bloom", action=argparse.BooleanOptionalAction, default=None, help="Toggle bloom."
    )
    look.add_argument("--bloom-strength", type=float, default=None, help="Bloom add-back.")
    look.add_argument("--bloom-threshold", type=float, default=None, help="Bloom luminance cut.")
    look.add_argument("--exposure", type=float, default=None, help="HDR exposure.")
    look.add_argument("--vignette", type=float, default=None, help="Corner darkening 0..1.")
    look.add_argument("--speed-lo", type=float, default=None, help="Speed at the cold end.")
    look.add_argument("--speed-hi", type=float, default=None, help="Speed at the hot end.")
    look.add_argument("--density-cell", type=float, default=None, help="DENSITY-mode cell side.")
    look.add_argument("--uniform-t", type=float, default=None, help="UNIFORM-mode LUT coord.")
    look.add_argument("--zoom", type=float, default=1.0, help="Fit-to-world zoom multiplier.")
    look.add_argument(
        "--follow",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Camera tracks the flock centre of mass.",
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
        start_frame=args.start_frame,
        max_frames=args.max_frames,
        auto_speed=not args.no_auto_speed,
        point_size=args.point_size,
        glow=args.glow,
        intensity=args.intensity,
        trail_decay=args.trail_decay,
        bloom=args.bloom,
        bloom_strength=args.bloom_strength,
        bloom_threshold=args.bloom_threshold,
        exposure=args.exposure,
        vignette=args.vignette,
        speed_lo=args.speed_lo,
        speed_hi=args.speed_hi,
        density_cell=args.density_cell,
        uniform_t=args.uniform_t,
        zoom=args.zoom,
        follow=args.follow,
    )
    engine = ReplayEngine(args.stream, cfg)
    engine.run()
    if args.export:
        print(f"Rendered {args.stream} -> {args.export}")
    return 0
