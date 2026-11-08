"""Fetch exact upstream commits into ignored local directories; never use floating branches."""

import json
import subprocess
import sys

from .paths import ROOT, UPSTREAM


def fetch_upstream(install=False):
    lock = json.loads((ROOT / "configs/upstream.lock.json").read_text())
    for name, entry in lock["repositories"].items():
        directory = UPSTREAM / name
        directory.parent.mkdir(parents=True, exist_ok=True)
        if not directory.exists():
            subprocess.run(["git", "clone", entry["url"], str(directory)], check=True)
        actual = subprocess.check_output(["git", "-C", str(directory), "rev-parse", "HEAD"], text=True).strip()
        if subprocess.check_output(["git", "-C", str(directory), "status", "--porcelain", "--untracked-files=no"], text=True):
            raise RuntimeError(f"Tracked upstream modifications in {directory}; cannot claim a locked version")
        if actual != entry["commit"]:
            if subprocess.check_output(["git", "-C", str(directory), "status", "--porcelain"], text=True):
                raise RuntimeError(f"Uncommitted upstream changes in {directory}; will not reset them")
            subprocess.run(["git", "-C", str(directory), "fetch", "origin", entry["commit"]], check=True)
            subprocess.run(["git", "-C", str(directory), "checkout", "--detach", entry["commit"]], check=True)
        print(f"{name}: {entry['commit']}")
    if install:
        packages = [UPSTREAM / "beyondAMP/source/beyondAMP", UPSTREAM / "beyondAMP/source/rsl_rl_amp",
                    UPSTREAM / "robotlib"]
        for package in packages:
            subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "-e", str(package)], check=True)
