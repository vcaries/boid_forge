"""Boid solvers: four interchangeable backends behind one interface.

* :class:`~boidforge.solver.naive.NaiveSolver` — L1, O(N²) reference.
* :class:`~boidforge.solver.vectorized.VectorizedSolver` — L2, O(N²) NumPy.
* :class:`~boidforge.solver.spatial_hash.SpatialHashSolver` — L3, uniform grid.
* :class:`~boidforge.solver.native.NativeSolver` — L4, CPython C extension.

All backends implement :class:`~boidforge.solver.base.Solver` and must produce
bit-identical results for the same ``(config, seed)`` (see ``CLAUDE.md`` §3).
This subpackage contains no visualization code.
"""

from __future__ import annotations

from boidforge.solver.base import Solver
from boidforge.solver.naive import NaiveSolver
from boidforge.solver.native import NativeSolver
from boidforge.solver.spatial_hash import SpatialHashSolver
from boidforge.solver.vectorized import VectorizedSolver

#: Registry mapping backend name -> solver class, for CLI/benchmark selection.
SOLVERS: dict[str, type[Solver]] = {
    NaiveSolver.name: NaiveSolver,
    VectorizedSolver.name: VectorizedSolver,
    SpatialHashSolver.name: SpatialHashSolver,
    NativeSolver.name: NativeSolver,
}

__all__ = [
    "Solver",
    "NaiveSolver",
    "VectorizedSolver",
    "SpatialHashSolver",
    "NativeSolver",
    "SOLVERS",
]
