"""Suite-wide test setup.

Real-footage fixtures are not distributed. The original author's tests read small motion and camera
files extracted from their own fight clips (``tests/motion/fixtures/*.npz``,
``tests/models/fixtures/*.npz``). Those clips are copyrighted, so the files are not in this repository.
A test that needs one is **skipped**, with the file's name, instead of failing. To run those tests,
regenerate the files from your own footage with ``tests/motion/fixtures/make_real_fixtures.py`` and
``tests/models/fixtures/make_gemx_fixtures.py`` (see AGENTS.md, section 4).
"""

from __future__ import annotations

from pathlib import Path

import pytest


def _missing_real_fixture(exc: BaseException | None) -> str | None:
    """The fixture file's name when ``exc`` (or what caused it) is a missing ``tests/**/fixtures/*.npz``."""
    seen: set[int] = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if isinstance(exc, FileNotFoundError) and exc.filename:
            p = Path(str(exc.filename))
            if p.suffix == ".npz" and p.parent.name == "fixtures" and "tests" in p.parts:
                return f"{p.parent.parent.name}/fixtures/{p.name}"
        exc = exc.__cause__ or exc.__context__
    return None


def _skip_without_fixture(exc: Exception) -> None:
    name = _missing_real_fixture(exc)
    if name is not None:
        pytest.skip(f"needs the real-footage fixture tests/{name}, which is not distributed "
                    "(regenerate it from your own footage: AGENTS.md section 4)")


@pytest.hookimpl(wrapper=True)
def pytest_runtest_setup(item):
    try:
        return (yield)
    except Exception as e:
        _skip_without_fixture(e)
        raise


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item):
    try:
        return (yield)
    except Exception as e:
        _skip_without_fixture(e)
        raise
