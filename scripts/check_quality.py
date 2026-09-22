"""Full runtime-package typing and deterministic tests in the locked development env."""

import importlib.metadata
from pathlib import Path
import platform
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    print(
        f"Runtime: {platform.python_implementation()} {platform.python_version()} "
        f"{platform.system()} {platform.machine()}",
        flush=True,
    )
    for package in ("mypy", "pytest", "ruff", "types-jsonschema"):
        print(f"{package}=={importlib.metadata.version(package)}", flush=True)
    commands = [
        [
            "ruff",
            "check",
            "--target-version",
            "py311",
            "--select",
            "E9,F63,F7,F82",
            "synaptic",
            "tests",
            "scripts",
        ],
        ["mypy", "--config-file", "pyproject.toml", "--no-incremental", "synaptic"],
        ["pytest", "tests", "-q", "-W", "error::RuntimeWarning", "--tb=short"],
    ]
    for command in commands:
        print("Gate: " + " ".join(command), flush=True)
        subprocess.run([sys.executable, "-m", *command], cwd=root, check=True)


if __name__ == "__main__":
    main()
