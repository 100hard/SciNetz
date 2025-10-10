"""Dependency alignment tests for Docker build performance."""

from __future__ import annotations

from pathlib import Path

import tomllib


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _read_requirements(path: Path) -> list[str]:
    """Load requirement strings from a file, ignoring comments and blanks."""

    return [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def test_requirements_match_pyproject_dependencies() -> None:
    """Ensure runtime dependencies stay in sync between pyproject and requirements."""

    pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())
    pyproject_deps = sorted(pyproject["project"]["dependencies"])

    base_requirements = _read_requirements(PROJECT_ROOT / "requirements" / "base.txt")
    assert sorted(base_requirements) == pyproject_deps


def test_dev_requirements_match_optional_group() -> None:
    """Ensure dev requirements mirror the optional dev extra for local installs."""

    pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())
    optional = pyproject["project"]["optional-dependencies"]["dev"]
    expected_dev = sorted(optional)

    dev_requirements = _read_requirements(PROJECT_ROOT / "requirements" / "dev.txt")
    assert dev_requirements[0] == "-r base.txt"
    assert sorted(dev_requirements[1:]) == expected_dev
