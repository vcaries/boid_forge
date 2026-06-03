# BoidForge — Architecture

## 1. System overview

BoidForge is split into two subsystems that never share runtime state, only an
on-disk binary contract:

```
┌──────────────────────┐     .bfs stream      ┌──────────────────────────┐
│   Simulation Solver   │  ───────────────►   │  Visualization Engine     │
│   (compute layer)     │   (binary frames)    │  (post-processing layer)  │
│                       │                      │                           │
│  physics + integration│                      │  read frames, render,     │
│  multi-backend kernels│                      │  camera, trails, video    │
│  writes snapshots      │                      │  NEVER simulates          │
└──────────────────────┘                      └──────────────────────────┘
```

The solver produces; the visualizer consumes. Each can run on a different
machine, at a different time. This separation is the central design constraint
(see `CLAUDE.md` §1) and exists to keep the compute path free of any rendering
dependency and to make every render a *deterministic replay* of recorded data.

### Package map

```
src/boidforge/
  core/        config, simulation state (SoA), shared types/constants
  io/          binary .bfs format: writer + reader (the cross-subsystem contract)
  solver/      L1 naive, L2 vectorized, L3 spatial-hash, L4 native — one interface
  benchmark/   timing harness + matplotlib plotting (scaling, speedup)
  viz/         replay engine, ModernGL renderer, camera, FFmpeg video export
native/        CPython C extension (SoA kernel) for L4, built via CMake
```

---

## 2. Solver pipeline (compute layer)

```
SimulationConfig ─► Solver.seed(state) ─► loop t in [0, steps):
                                            Solver.step(state, dt)
                                            FrameWriter.write(t, state)
                                          ─► FrameWriter.close()
```

The solver owns a `SimulationState` (Struct-of-Arrays: `px, py, vx, vy`,
`float32`, length `N`). Each step applies the three boid rules and integrates:

- **Separation** — steer away from neighbors closer than `r_sep`.
- **Alignment** — steer toward the mean heading of neighbors within `r_ali`.
- **Cohesion** — steer toward the mean position of neighbors within `r_coh`.

then clamps speed to `[min_speed, max_speed]`, integrates position by `dt`, and
applies boundary behavior (wrap or reflect, per config).

Four interchangeable backends implement the identical update:

| Level | Module                          | Neighbor search        | Complexity |
|-------|---------------------------------|------------------------|------------|
| L1    | `solver/naive.py`               | all-pairs (Python loop)| O(N²)      |
| L2    | `solver/vectorized.py`          | all-pairs (NumPy)      | O(N²)      |
| L3    | `solver/spatial_hash.py`        | uniform grid buckets   | ~O(N·k)    |
| L4    | `solver/native.py` → `_native`  | uniform grid in C      | ~O(N·k)    |

L1 is the correctness reference. L2/L3/L4 must match it bit-for-bit under the
determinism contract (§6). L2 is the same all-pairs algorithm as L1 re-expressed
with NumPy broadcasting — it isolates the speedup attributable to vectorization
alone, at the cost of O(N²) memory for the pairwise matrices. The C extension is
the production target for 10k–100k boids.

The solver layer imports nothing from `viz` and contains no rendering logic.

---

## 3. Visualization pipeline (post-processing layer)

```
.bfs file ─► FrameReader (sequential) ─► ReplayEngine ─► Renderer (ModernGL)
                                                       ├► Camera (follow COM, zoom/pan)
                                                       └► VideoExporter (FFmpeg) [optional]
```

The visualizer opens a `.bfs` stream and pulls frames in order. For each frame
it uploads positions/velocities to the GPU and draws instanced point sprites.
Velocity magnitude drives color mapping; an accumulation buffer produces motion
trails; the camera can track the center of mass with zoom/pan. Output is either
an interactive window or, with `VideoExporter`, an H.264 file via FFmpeg.

The visualizer imports nothing from `solver`/`_native` and never advances
physics. If a frame is missing, it is a data problem, not something to compute.

---

## 4. Binary format `.bfs` — specification

> Magic `BFS1`, little-endian, IEEE-754 `float32` payloads, component-major
> (SoA) layout.

### 4.1 Why this design (justification — read before changing)

Requirements: fast sequential **write**, fast sequential **read**, minimal disk
overhead, deterministic replay. The design follows directly from those:

- **Component-major (SoA) payload, not interleaved.** The solver state is four
  separate contiguous arrays `px, py, vx, vy`. Writing them back-to-back lets
  the writer emit each array with a single `ndarray.tofile`/`fwrite` — no
  per-boid repacking, no temporary interleave buffer. The reader maps each
  array straight into a NumPy view that the renderer can upload as-is. AoS
  (`x0,y0,vx0,vy0,…`) would force a transpose on both ends. SoA also matches the
  cache-friendly iteration order of every solver backend.
- **Flat append-only stream, frame = tiny header + raw arrays.** Per-frame
  metadata is 8 bytes (`timestep`, `n_boids`); everything else is bulk payload.
  This maximizes the ratio of useful bytes to overhead and makes writing a pure
  sequential append (ideal for the OS page cache and SSD streaming). No
  per-frame checksums or JSON — metadata that would bloat the stream lives once
  in the global header.
- **Fixed endianness + fixed dtype, declared once in the global header.** A
  reader validates the header and then trusts the layout; replay is byte-exact
  across machines. Endianness/dtype are explicit so the format is portable
  rather than "whatever the writer's CPU was".
- **`n_boids` per frame.** Costs 4 bytes/frame but allows populations to change
  over time and lets a reader length-check each frame independently — cheap
  insurance for streaming robustness.
- **Optional sidecar index `.bfx`, not inline offsets.** Random seek (jump to
  frame *k*) is a *visualizer* concern, not a *writer* concern. Forcing the
  writer to maintain an offset table would add seeks/bookkeeping to the hot
  write path. Instead the writer stays append-only; a separate, optional pass
  builds a `.bfx` offset index for scrubbing. Sequential replay needs no index.

Net: the write path is "append small header, dump four arrays"; the read path is
"read header, `np.fromfile` four arrays into views". Both are O(payload) with no
copies or transposes.

### 4.2 Global header (32 bytes, written once)

| Offset | Field          | Type      | Notes                                    |
|--------|----------------|-----------|------------------------------------------|
| 0      | `magic`        | `char[4]` | `"BFS1"` (0x42 46 53 31)                 |
| 4      | `version`      | `uint16`  | `FORMAT_VERSION`, currently `1`          |
| 6      | `flags`        | `uint16`  | bit0 = little-endian (1), others reserved|
| 8      | `dim`          | `uint8`   | spatial dimensions, `2`                  |
| 9      | `dtype_code`   | `uint8`   | `0` = float32 (only value in v1)         |
| 10     | `reserved`     | `uint8[2]`| zero                                     |
| 12     | `max_boids`    | `int32`   | upper bound for buffers; `0` = unbounded |
| 16     | `dt`           | `float32` | timestep used by the solver              |
| 20     | `seed`         | `uint32`  | RNG seed (provenance / reproducibility)  |
| 24     | `frame_count`  | `int32`   | filled on `close()`; `-1` while streaming|
| 28     | `reserved2`    | `uint8[4]`| zero (pads header to 32 bytes)           |

### 4.3 Frame record (repeated, variable length)

For a frame with `N = n_boids`:

| Field        | Type            | Bytes      |
|--------------|-----------------|------------|
| `timestep`   | `int32`         | 4          |
| `n_boids`    | `int32`         | 4          |
| `px`         | `float32[N]`    | 4N         |
| `py`         | `float32[N]`    | 4N         |
| `vx`         | `float32[N]`    | 4N         |
| `vy`         | `float32[N]`    | 4N         |

Frame size = `8 + 16·N` bytes. Records are concatenated with no padding. End of
file = end of stream.

### 4.4 Optional index `.bfx`

A flat array of `int64` byte offsets, one per frame, pointing at each frame
header within the `.bfs`. Built by an offline pass; enables O(1) seek for
scrubbing. Absent by default; sequential replay does not require it.

---

## 5. Module boundaries

| Module                       | May import                       | Must NOT import                 |
|------------------------------|----------------------------------|---------------------------------|
| `core/`                      | stdlib, numpy                    | solver, viz, io                 |
| `io/`                        | stdlib, numpy, core              | solver, viz                     |
| `solver/`                    | core, io (writer), numpy, `_native` | viz, moderngl, matplotlib    |
| `benchmark/`                 | core, solver, io, matplotlib     | viz                             |
| `viz/`                       | core, io (reader), moderngl      | solver, `_native`               |
| `native/` (`_native`)        | CPython C-API, numpy C-API       | anything file/render related    |

`io/` is the shared contract and depends only on `core/`, so both subsystems can
import it without coupling to each other. The layering guard test
(`tests/test_layering.py`) asserts these rules statically.

---

## 6. Determinism contract

Identical `(SimulationConfig, seed)` ⇒ identical `.bfs` bytes across all
backends and across machines. To hold this:

- All state and arithmetic in `float32`; reductions accumulated in a fixed
  order. Neighbor contributions are summed in ascending boid-index order so
  floating-point non-associativity cannot reorder results between backends.
  This constrains vectorization: each boid's reduction is a `np.sum`/`np.mean`
  over its *compacted* neighbor list, never a reduction over a zero-padded
  pairwise row (which would regroup the additions and diverge in the last ULP).
  L2 therefore vectorizes the geometry and clamps but keeps the per-boid
  reduction loop.
- RNG is seeded solely from `config.seed`; initial positions/velocities are a
  pure function of the seed and `N`.
- No wall-clock, hostname, hash-seed, set/dict iteration, or thread-scheduling
  influence on results. Any parallelism uses deterministic partitioning.
- Boundary handling, speed clamping, and rule weights are config-driven
  constants, not literals scattered in backends.

`tests/test_solvers_equivalence.py` runs each implemented backend for several
`(N, seed)` and asserts identical output frames.

---

## 7. Benchmarking

`benchmark/runner.py` measures, per backend and per `N`:

- per-frame execution time (ms) and FPS-equivalent (`1000 / ms`),
- scaling of time vs `N`,
- speedup ratios relative to the L1 baseline (L1→L2→L3→L4).

Results export to a structured file (CSV/JSON); `benchmark/plots.py` renders
matplotlib figures (time-vs-N on log axes, speedup bars). Benchmark artifacts
are reproducible and are **not** committed (`.gitignore`).
