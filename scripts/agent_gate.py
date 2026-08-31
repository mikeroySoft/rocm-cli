#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Deterministic quality gate for agent worktrees.

Runs from a worktree root. Exit 0 = all checks pass, 1 = any failure.
Writes a short Markdown report (PASS/FAIL per check + failure excerpts).
"""

from __future__ import annotations

import argparse
import fcntl
import re
import subprocess
import sys
from pathlib import Path

GPU_LOCK = "/tmp/rocm-factory-gpu.lock"
LEAK_RE = re.compile(
    r"internal|confidential|proprietary|private|jira|confluence|\.corp|\.internal",
    re.IGNORECASE,
)
TAIL_LINES = 40


def run(cmd: list[str]) -> tuple[bool, str]:
    """Run a command, return (passed, combined output)."""
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode == 0, proc.stdout + proc.stderr


def check_conflict_markers() -> tuple[bool, str]:
    proc = subprocess.run(
        # Exactly-7-char markers only; long ===== separator lines are legit.
        ["git", "grep", "-nE", r"^(<{7}|={7}|>{7})( |$)"],
        capture_output=True,
        text=True,
    )
    # git grep exits 1 when nothing matches — that is the pass case.
    if proc.returncode == 1:
        return True, ""
    return False, proc.stdout + proc.stderr


def check_clippy() -> tuple[bool, str]:
    return run(["cargo", "clippy", "--workspace", "--all-targets", "--", "-D", "warnings"])


def check_tests() -> tuple[bool, str]:
    return run(["cargo", "test", "--workspace", "--all-targets"])


def check_smoke() -> tuple[bool, str]:
    return run([sys.executable, "scripts/smoke_local.py"])


def check_leaks(base: str) -> tuple[bool, str]:
    # ponytail: excludes fork-local agent-harness paths that never go upstream;
    # tighten the pathspec if those dirs ever feed a push.
    proc = subprocess.run(
        [
            "git",
            "diff",
            f"{base}..HEAD",
            "--",
            ":(exclude).agents",
            ":(exclude)skills",
            ":(exclude)skills-lock.json",
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return False, proc.stdout + proc.stderr
    hits = [
        line
        for line in proc.stdout.splitlines()
        if line.startswith("+") and not line.startswith("+++") and LEAK_RE.search(line)
    ]
    return not hits, "\n".join(hits)


GPU_CHECKS = {"clippy", "tests", "smoke"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Quality gate for agent worktrees.")
    parser.add_argument("--base", default="origin/main", help="base ref for leak scan")
    parser.add_argument(
        "--report", default=".factory/gate-report.md", help="report output path"
    )
    parser.add_argument(
        "--skip",
        default="",
        help="comma-separated check names to skip "
        "(conflict-markers, clippy, tests, smoke, leak-scan)",
    )
    args = parser.parse_args()
    skip = {name.strip() for name in args.skip.split(",") if name.strip()}

    checks = [
        ("conflict-markers", check_conflict_markers),
        ("clippy", check_clippy),
        ("tests", check_tests),
        ("smoke", check_smoke),
        ("leak-scan", lambda: check_leaks(args.base)),
    ]

    results: list[tuple[str, str, str]] = []  # (name, status, output)
    gpu_lock = None
    for name, fn in checks:
        if name in skip:
            results.append((name, "SKIP", ""))
            continue
        if name in GPU_CHECKS and gpu_lock is None:
            gpu_lock = open(GPU_LOCK, "w")
            fcntl.flock(gpu_lock, fcntl.LOCK_EX)
        passed, output = fn()
        results.append((name, "PASS" if passed else "FAIL", output))
    if gpu_lock is not None:
        fcntl.flock(gpu_lock, fcntl.LOCK_UN)
        gpu_lock.close()

    lines = ["# Gate report", ""]
    for name, status, _ in results:
        lines.append(f"- {name}: {status}")
    for name, status, output in results:
        if status == "FAIL":
            tail = "\n".join(output.splitlines()[-TAIL_LINES:])
            lines += ["", f"## {name} failure", "```", tail, "```"]
    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n")

    failed = [name for name, status, _ in results if status == "FAIL"]
    summary = "gate FAIL: " + ", ".join(failed) if failed else "gate PASS"
    print(f"report: {report}")
    print(summary)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
