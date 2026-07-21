"""COMET: Comprehensive Orbit Mechanics and Engineering Toolkit

A Python library for orbital mechanics, satellite operations, and space mission analysis.
"""

from ._version import version as __version__

import dataclasses
from dataclasses import make_dataclass
from pathlib import Path


def _discover_module_paths(code: Path) -> dict[str, Path]:
    """Return top-level subdirectories of the package root as a name→Path mapping.

    Skips hidden directories and Python tooling directories such as
    ``__pycache__``, ``*.dist-info``, and ``*.egg-info``.

    Args:
        code (Path): The package root (directory containing ``__init__.py``).

    Returns:
        dict[str, Path]: Mapping of uppercased directory name to its absolute Path,
            e.g. ``{"DATA": Path(".../data"), "CORE": Path(".../core")}``.
    """
    _SKIP = {"__pycache__"}

    def _is_skipped(p: Path) -> bool:
        n = p.name
        return n.startswith(".") or n.endswith((".dist-info", ".egg-info")) or n in _SKIP

    return {d.name.upper(): d for d in sorted(code.iterdir()) if d.is_dir() and not _is_skipped(d)}


def _make_paths(repo: Path, code: Path) -> object:
    """Construct a frozen dataclass instance holding all project paths.

    The fixed fields ``REPO`` and ``CODE`` are always present. Additional
    fields are auto-discovered from the top-level subdirectories of ``code``
    and added dynamically via :func:`dataclasses.make_dataclass`.

    Args:
        repo (Path): Absolute path to the repository root.
        code (Path): Absolute path to the package root (``src/projectname/``).

    Returns:
        object: A frozen dataclass instance with ``REPO``, ``CODE``, and one
            uppercased field per discovered top-level module directory.
    """
    modules = _discover_module_paths(code)
    fields = [
        ("REPO", Path),
        ("CODE", Path),
        *((name, Path) for name in modules),
    ]

    def __repr__(self) -> str:
        lines = [f"  {f.name} = {getattr(self, f.name)}" for f in dataclasses.fields(self)]
        return "ProjectPaths(\n" + "\n".join(lines) + "\n)"

    Paths = make_dataclass("Paths", fields, frozen=True, namespace={"__repr__": __repr__})
    return Paths(repo, code, **modules)


_CODE = Path(__file__).resolve().parent
_REPO = _CODE.parents[1]
PATHS = _make_paths(_REPO, _CODE)

__all__ = ["__version__", "PATHS"]
