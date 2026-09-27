#!/usr/bin/env python3
"""Build the python module.

Wraps the PyPA build backend so a release is one command from a clean tree:

    python build.py                 # clean, then build sdist + wheel into dist/
    python build.py --test          # run pytest first, abort if it fails
    python build.py --wheel         # wheel only (sdist only: --sdist)
    python build.py --check         # twine check the artifacts afterwards
    python build.py --install       # pip install the wheel it just built
    python build.py --no-clean      # keep whatever is already in dist/
    python build.py --clean-only    # remove build/, dist/, *.egg-info, caches
    python build.py --no-isolation  # build in this env, no pip download

Everything runs against the interpreter that launched this script, so building
inside a venv builds for that venv.

Note: this file is named ``build.py``, so it shadows the PyPA ``build`` package
for anything run from the project root. Every subprocess here is therefore
launched from a neutral directory with the project passed as an explicit source
path -- do not "simplify" that back to a plain ``python -m build`` in ROOT.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DIST = ROOT / "dist"

# Any cwd but ROOT, so `import build` finds PyPA build instead of this file.
NEUTRAL = Path(tempfile.gettempdir())

# Directories wiped by --clean (the default); globs are resolved under ROOT.
CLEAN_GLOBS = ("build", "dist", "*.egg-info", ".pytest_cache", "**/__pycache__")


def run(cmd: list[str], *, what: str, cwd: Path = ROOT) -> None:
    """Run a subprocess, exiting on failure."""
    print(f"\n==> {what}\n    {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        sys.exit(f"\n{what} failed (exit {result.returncode})")


def have_module(name: str) -> bool:
    return (
        subprocess.run(
            [sys.executable, "-c", f"import {name}"],
            cwd=NEUTRAL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        == 0
    )


def require_module(name: str, pip_name: str | None = None) -> None:
    if have_module(name):
        return
    sys.exit(
        f"missing build dependency: {name}\n"
        f"    {sys.executable} -m pip install {pip_name or name}"
    )


def clean() -> None:
    print("\n==> clean", flush=True)
    removed = 0
    for pattern in CLEAN_GLOBS:
        for path in sorted(ROOT.glob(pattern), reverse=True):
            if not path.exists():
                continue
            print(f"    rm {path.relative_to(ROOT)}")
            shutil.rmtree(path) if path.is_dir() else path.unlink()
            removed += 1
    if not removed:
        print("    nothing to remove")


def test() -> None:
    require_module("pytest", 'pytest   # or: pip install -e ".[dev]"')
    run([sys.executable, "-m", "pytest", "-q"], what="tests")


def build(sdist: bool, wheel: bool, isolated: bool = True) -> None:
    require_module("build")
    cmd = [sys.executable, "-m", "build", "--outdir", str(DIST)]
    if sdist != wheel:  # one of the two; passing neither flag means both
        cmd.append("--sdist" if sdist else "--wheel")
    if not isolated:
        cmd.append("--no-isolation")
    cmd.append(str(ROOT))
    run(cmd, what="build", cwd=NEUTRAL)


def check() -> None:
    require_module("twine")
    run(
        [sys.executable, "-m", "twine", "check", *map(str, artifacts())],
        what="twine check",
        cwd=NEUTRAL,
    )


def artifacts() -> list[Path]:
    return sorted(DIST.glob("*.whl")) + sorted(DIST.glob("*.tar.gz"))


def install() -> None:
    wheels = sorted(DIST.glob("*.whl"))
    if not wheels:
        sys.exit("no wheel in dist/ to install")
    newest = max(wheels, key=lambda p: p.stat().st_mtime)
    run(
        [sys.executable, "-m", "pip", "install", "--force-reinstall", str(newest)],
        what=f"install {newest.name}",
        cwd=NEUTRAL,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the distribution.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--sdist", action="store_true", help="build the sdist only")
    parser.add_argument("--wheel", action="store_true", help="build the wheel only")
    parser.add_argument("--test", action="store_true", help="run pytest before building")
    parser.add_argument("--check", action="store_true", help="twine check the artifacts")
    parser.add_argument(
        "--install", action="store_true", help="pip install the built wheel afterwards"
    )
    parser.add_argument(
        "--no-clean",
        dest="clean",
        action="store_false",
        help="keep build/ and dist/ as they are",
    )
    parser.add_argument(
        "--clean-only", action="store_true", help="clean and exit without building"
    )
    parser.add_argument(
        "--no-isolation",
        dest="isolated",
        action="store_false",
        help="build in the current environment instead of a fresh one (offline)",
    )
    args = parser.parse_args(argv)

    if args.clean_only:
        clean()
        return 0

    if args.test:
        test()
    if args.clean:
        clean()
    build(sdist=args.sdist, wheel=args.wheel, isolated=args.isolated)

    made = artifacts()
    print("\n==> built")
    for path in made:
        print(f"    dist/{path.name}  ({path.stat().st_size / 1024:.0f} KiB)")
    if not made:
        sys.exit("build produced no artifacts in dist/")

    if args.check:
        check()
    if args.install or not args.sdist:  # wheel is the default, so install if no sdist-only
        install()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
