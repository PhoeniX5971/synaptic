"""Install each declared extra into an isolated temporary venv; no provider requests."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile

EXTRAS = {
    "core": None,
    "mcp": "synaptic.mcp",
    "openai": "synaptic.providers.openai_",
    "deepseek": "synaptic.providers.deepseek",
    "claude": "synaptic.providers.claude",
    "gemini": "synaptic.providers.gemini",
    "vertex": "synaptic.providers.vertex",
    "together": "synaptic.providers.together",
    "xai": "synaptic.providers.xai",
    "all": "synaptic.providers",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("--constraints", type=Path, required=True)
    parser.add_argument("--python", default="3.13")
    args = parser.parse_args()
    environment = {
        **os.environ,
        "UV_CACHE_DIR": os.environ.get("UV_CACHE_DIR", "/tmp/synaptic-uv-cache"),
    }
    with tempfile.TemporaryDirectory(prefix="synaptic-installs-") as temporary:
        for extra, module in EXTRAS.items():
            directory = Path(temporary) / extra
            subprocess.run(
                ["uv", "venv", "--python", args.python, str(directory)],
                check=True,
                env=environment,
                stdout=subprocess.DEVNULL,
            )
            python = directory / "bin/python"
            requirement = str(args.wheel.resolve()) + (f"[{extra}]" if extra != "core" else "")
            subprocess.run(
                [
                    "uv",
                    "pip",
                    "install",
                    "--python",
                    str(python),
                    "--constraint",
                    str(args.constraints.resolve()),
                    requirement,
                ],
                check=True,
                env=environment,
                stdout=subprocess.DEVNULL,
            )
            check = """import importlib, importlib.util, sys
import synaptic
assert not any(name in sys.modules for name in ('openai', 'anthropic', 'google.genai', 'xai_sdk', 'together'))
"""
            if module:
                check += f"importlib.import_module({module!r})\n"
            if extra == "all":
                check += "from synaptic.providers import *\n"
            if extra == "core":
                check += "assert importlib.util.find_spec('openai') is None\n"
                check += "try:\n from synaptic import OpenAIAdapter\nexcept ImportError as error:\n assert 'synaptic[openai]' in str(error)\nelse:\n raise AssertionError('missing SDK not reported')\n"
            subprocess.run(
                [str(python), "-I", "-c", check], check=True, cwd=temporary, env=environment
            )
            subprocess.run(
                ["uv", "pip", "check", "--python", str(python)],
                check=True,
                env=environment,
                stdout=subprocess.DEVNULL,
            )
            print(
                json.dumps({"extra": extra, "status": "passed", "python": args.python}), flush=True
            )


if __name__ == "__main__":
    main()
