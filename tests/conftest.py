import importlib.util
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load_package():
    if "panda3d_vr" in sys.modules:
        return sys.modules["panda3d_vr"]
    spec = importlib.util.spec_from_file_location(
        "panda3d_vr", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules["panda3d_vr"] = mod
    spec.loader.exec_module(mod)
    return mod


_load_package()


@pytest.fixture(scope="session")
def base():
    """One offscreen ShowBase for the whole run (Panda allows only one)."""
    from panda3d.core import loadPrcFileData
    from direct.showbase.ShowBase import ShowBase

    loadPrcFileData("tests", "window-type offscreen\nsync-video false\naudio-library-name null\n")
    if sys.platform.startswith("linux"):
        # The XR session binds to an EGL context on Linux (see _gl.py).
        loadPrcFileData("tests", "load-display p3headlessgl\n")
    b = ShowBase()
    yield b
    b.destroy()
