"""Static guard enforcing the two-subsystem import boundary (CLAUDE.md §1).

Parses every module in ``src/boidforge`` and asserts that no subpackage imports
across a forbidden boundary. This keeps the compute layer free of rendering
dependencies and the visualization layer free of solver/kernel coupling. Do not
weaken these rules.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_PKG_ROOT = Path(__file__).resolve().parents[1] / "src" / "boidforge"

# subpackage relative dir -> tuple of forbidden import prefixes
_RULES: dict[str, tuple[str, ...]] = {
    "core": ("boidforge.solver", "boidforge.io", "boidforge.viz", "boidforge._native"),
    "io": ("boidforge.solver", "boidforge.viz"),
    "solver": ("boidforge.viz", "moderngl", "pyglet", "matplotlib"),
    "viz": ("boidforge.solver", "boidforge._native"),
    "benchmark": ("boidforge.viz", "moderngl", "pyglet"),
}


def _imported_modules(source: str) -> set[str]:
    """Collect fully-qualified module names imported anywhere in a source file."""
    tree = ast.parse(source)
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
    return names


def _python_files(subdir: str) -> list[Path]:
    """All .py files within a subpackage."""
    return sorted((_PKG_ROOT / subdir).rglob("*.py"))


@pytest.mark.parametrize("subdir", sorted(_RULES))
def test_no_forbidden_imports(subdir: str) -> None:
    """No module in ``subdir`` imports a forbidden boundary-crossing module."""
    forbidden = _RULES[subdir]
    for path in _python_files(subdir):
        imports = _imported_modules(path.read_text(encoding="utf-8"))
        for imported in imports:
            for bad in forbidden:
                assert not (
                    imported == bad or imported.startswith(bad + ".")
                ), f"{path.relative_to(_PKG_ROOT)} illegally imports {imported}"
