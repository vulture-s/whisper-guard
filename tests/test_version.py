"""The version the package reports must be the version it was published as.

v0.3.0 went to PyPI with `Version: 0.3.0` in its metadata and
`__version__ = "0.2.0"` inside the wheel: the number was kept in two places,
one of them was bumped, and nothing compared them. `pip show whisper-guard`
and `whisper_guard.__version__` disagreed for two months.

The build now reads the version from `__init__.py` (hatchling
`[tool.hatch.version]`), so the two cannot diverge by construction. This test
is the belt: it fails if anyone reintroduces a second source, and — because it
compares against *installed metadata* — it also catches a wheel built from a
stale tree.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import whisper_guard


def test_installed_metadata_matches_dunder_version():
    from importlib.metadata import PackageNotFoundError, version

    try:
        installed = version("whisper-guard")
    except PackageNotFoundError:
        pytest.skip("whisper-guard is not installed in this environment")

    assert installed == whisper_guard.__version__, (
        "installed metadata says %s but the package reports %s — "
        "the wheel was built from a different tree than it claims"
        % (installed, whisper_guard.__version__)
    )


def test_pyproject_declares_version_dynamic_not_literal():
    """A literal `version = "..."` in pyproject is the second source that
    caused the drift; keep it gone."""
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    if not pyproject.exists():
        pytest.skip("pyproject.toml not present (installed-only environment)")
    text = pyproject.read_text(encoding="utf-8")

    literal = re.search(r"^version\s*=\s*[\"']", text, re.M)
    assert literal is None, "pyproject.toml pins a literal version again"
    assert 'dynamic = ["version"]' in text
    assert "[tool.hatch.version]" in text


def test_version_is_pep440_sortable():
    assert re.fullmatch(r"\d+\.\d+\.\d+([abrc].*)?", whisper_guard.__version__)
