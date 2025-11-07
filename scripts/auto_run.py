#!/usr/bin/env python3
"""
Local automation runner for common dev tasks.

Usage examples:
  # Install deps and editable install
  python scripts/auto_run.py setup

  # Run lint and tests
  python scripts/auto_run.py check

  # Start API (foreground)
  python scripts/auto_run.py api

  # Run full pipeline: setup -> lint -> tests
  python scripts/auto_run.py full

This small script is intentionally simple and uses subprocess to run
the same commands you'd run in the terminal, making it cross-platform.
"""

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(cmd, check=True):
    print(f"$ {cmd}")
    res = subprocess.run(cmd, shell=True, cwd=ROOT)
    if check and res.returncode != 0:
        print(f"Command failed: {cmd} (exit {res.returncode})")
        sys.exit(res.returncode)
    return res.returncode


def cmd_setup():
    run("python -m pip install --upgrade pip setuptools wheel")
    run("python -m pip install -r requirements.txt")
    # editable install may fail in flat layout; attempt but don't fail hard
    run("python -m pip install -e . || true", check=False)


def cmd_check():
    run("python -m ruff check .")
    run("pytest -q")


def cmd_api():
    run("uvicorn api.main:app --reload")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["setup", "check", "api", "full"], help="Action to run")
    args = p.parse_args()

    if args.action == "setup":
        cmd_setup()
    elif args.action == "check":
        cmd_check()
    elif args.action == "api":
        cmd_api()
    elif args.action == "full":
        cmd_setup()
        cmd_check()


if __name__ == "__main__":
    main()
