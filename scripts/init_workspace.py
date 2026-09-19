"""Create ignored, local-only research directories. Never downloads data."""

from pathlib import Path

root = Path(__file__).resolve().parents[1]
for relative in ("data/raw", "data/processed", "data/manifests/local", "runs"):
    path = root / relative
    path.mkdir(parents=True, exist_ok=True)
    print(path)
