# BoidForge

High-performance 2D Boids simulation and replay system, engineered as two
strictly decoupled subsystems connected by an on-disk binary stream.

```
Solver (compute)  ──►  .bfs binary stream  ──►  Visualization engine (replay)  ──►  frames / video
```

The **solver** computes boid physics and writes timestep snapshots to disk. It
contains **no rendering logic**. The **visualization engine** reads precomputed
snapshots and renders them; it **never runs the simulation**. The two halves
share only the binary format contract defined in
[`docs/architecture.md`](docs/architecture.md).

## Solver levels

| Level | Backend                          | Complexity | Target scale  |
|-------|----------------------------------|------------|---------------|
| L1    | Naive Python (all-pairs loop)    | O(N²)      | ≤ 2k boids    |
| L2    | Vectorized NumPy (all-pairs)     | O(N²)      | ≤ 2k boids    |
| L3    | Uniform-grid spatial hash (Py)   | ~O(N)      | ≤ 20k boids   |
| L4    | CPython C extension (SoA kernel) | ~O(N)      | 10k–100k boids|

All levels are required to produce **bit-reproducible** results for identical
configuration and seed (see the determinism contract in `docs/architecture.md`).

## Status

Scaffold + skeleton. Module interfaces and the binary format are defined;
physics kernels, the renderer, and benchmarks are not yet implemented. Bodies
intentionally raise `NotImplementedError` rather than contain placeholder logic.

## Layout

```
src/boidforge/      Python package (core, io, solver, benchmark, viz)
native/             CPython C extension (SoA kernel), built with CMake
docs/architecture.md  Single source of truth for the design + binary format
scripts/            CLI entry points (simulate, benchmark, replay)
tests/              Correctness + cross-level equivalence tests
CLAUDE.md           Operating rules for AI-assisted work in this repo
```

## Build (developer)

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e .            # builds the C extension via the CMake backend
```

Requires a C compiler, CMake ≥ 3.18, and Python ≥ 3.10 development headers.
