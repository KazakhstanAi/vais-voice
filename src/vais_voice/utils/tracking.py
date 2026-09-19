"""Capture configuration and environment without recording environment-variable secrets."""

import importlib.metadata
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def snapshot(project_root: Path) -> dict[str, Any]:
    def git(*args: str) -> str | None:
        try:
            return subprocess.check_output(
                ["git", "-C", str(project_root), *args],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=10,
            ).strip()
        except (OSError, subprocess.SubprocessError):
            return None

    dirty = git("status", "--porcelain")
    return {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": None if dirty is None else bool(dirty),
        "packages": dict(
            sorted(
                (dist.metadata["Name"], dist.version)
                for dist in importlib.metadata.distributions()
                if dist.metadata["Name"]
            )
        ),
    }
