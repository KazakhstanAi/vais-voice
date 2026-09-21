"""Run one pinned VoxCPM 1.5 + Kazakh LoRA synthesis without fallback."""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--lora", type=Path, required=True)
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--cfg-value", type=float, required=True)
    parser.add_argument("--inference-timesteps", type=int, required=True)
    parser.add_argument("--max-len", type=int, required=True)
    parser.add_argument("--expected-lora-keys", type=int)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("The pinned VoxCPM pilot requires CUDA")
    sys.path.insert(0, str(args.upstream_root.resolve()))
    from voxcpm.core import VoxCPM  # noqa: PLC0415
    from voxcpm.model.voxcpm import LoRAConfig  # noqa: PLC0415

    lora_metadata = json.loads((args.lora / "lora_config.json").read_text(encoding="utf-8"))
    lora_config = LoRAConfig(**lora_metadata["lora_config"])
    random.seed(args.seed)
    np.random.seed(args.seed % (2**32))
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    model = VoxCPM.from_pretrained(
        hf_model_id=str(args.base_model.resolve()),
        load_denoiser=False,
        optimize=False,
        lora_config=lora_config,
        lora_weights_path=str(args.lora.resolve()),
    )
    loaded_keys, skipped_keys = model.tts_model.load_lora_weights(str(args.lora.resolve()))
    if not loaded_keys or skipped_keys:
        raise RuntimeError(
            f"LoRA verification failed: loaded={len(loaded_keys)}, skipped={len(skipped_keys)}"
        )
    if args.expected_lora_keys is not None and len(loaded_keys) != args.expected_lora_keys:
        raise RuntimeError(
            f"LoRA key count mismatch: expected {args.expected_lora_keys}, got {len(loaded_keys)}"
        )

    audio = model.generate(
        text=args.input_file.read_text(encoding="utf-8").strip(),
        prompt_wav_path=None,
        prompt_text=None,
        cfg_value=args.cfg_value,
        inference_timesteps=args.inference_timesteps,
        max_len=args.max_len,
        normalize=False,
        denoise=False,
    )
    sample_rate = int(model.tts_model.sample_rate)
    sf.write(args.output_file, audio, sample_rate, subtype="PCM_16")
    print(
        json.dumps(
            {
                "cuda": torch.version.cuda,
                "device": torch.cuda.get_device_name(),
                "loaded_lora_keys": len(loaded_keys),
                "sample_rate": sample_rate,
                "torch_version": torch.__version__,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
