#!/usr/bin/env python3
from __future__ import annotations

import os
import re
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - for Python < 3.11
    import tomli as tomllib

from packaging.version import parse


def read_version_from_pyproject(pyproject_path: Path) -> str:
    pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    version = pyproject.get("project", {}).get("version")
    if not version:
        raise SystemExit("No version found in pyproject.toml.")
    return version


def main() -> int:
    tag = os.environ["TAG"]
    print(f"Target Tag: {tag}")

    version = read_version_from_pyproject(Path("pyproject.toml"))
    print(f"Detected Package Version: {version}")

    if parse(tag) != parse(version):
        raise SystemExit(f"Tag {tag} does not match package version {version}.")

    print("Version check passed!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
