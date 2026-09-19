"""Inspect available dependencies without downloading weights or starting training."""

import importlib.util
import json
import shutil

from vais_voice.utils.tracking import snapshot


def main() -> None:
    from pathlib import Path

    report = snapshot(Path.cwd())
    report["ffmpeg_available"] = shutil.which("ffmpeg") is not None
    report["pytorch_installed"] = importlib.util.find_spec("torch") is not None
    if report["pytorch_installed"]:
        import torch

        report["torch_version"] = torch.__version__
        report["cuda_runtime"] = torch.version.cuda
        report["cuda_available"] = torch.cuda.is_available()
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
