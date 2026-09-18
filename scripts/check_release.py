"""Build and qualify distributions without publishing; run after check_quality.py."""

import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import zipfile


def main():
    root = Path(__file__).resolve().parents[1]
    env = {
        **os.environ,
        "UV_CACHE_DIR": os.environ.get("UV_CACHE_DIR", "/tmp/synaptic-uv-cache"),
    }

    def run(*arguments):
        subprocess.run(list(arguments), cwd=root, env=env, check=True)

    with tempfile.TemporaryDirectory(prefix="synaptic-release-") as temporary:
        directory = Path(temporary)
        run("uv", "lock", "--check")
        run("uv", "build", "--out-dir", temporary)
        (wheel,) = directory.glob("*.whl")
        (source,) = directory.glob("*.tar.gz")
        with zipfile.ZipFile(wheel) as archive:
            assert "synaptic/py.typed" in archive.namelist()
        with tarfile.open(source) as archive:
            assert any(name.endswith("/synaptic/py.typed") for name in archive.getnames())
        constraints = directory / "constraints.txt"
        run(
            "uv",
            "export",
            "--frozen",
            "--all-extras",
            "--no-emit-project",
            "--format",
            "requirements-txt",
            "--output-file",
            str(constraints),
        )
        run(
            sys.executable,
            "scripts/check_installs.py",
            str(wheel),
            "--constraints",
            str(constraints),
            "--python",
            sys.executable,
        )
        print("Wheel/sdist typing metadata and all isolated install gates passed.", flush=True)


if __name__ == "__main__":
    main()
