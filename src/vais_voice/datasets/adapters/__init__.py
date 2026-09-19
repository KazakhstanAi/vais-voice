"""Registry of local-only external dataset adapters."""

from pathlib import Path

from vais_voice.datasets.adapters.asvspoof2021 import ASVspoof2021Adapter
from vais_voice.datasets.adapters.base import DatasetAdapter, load_source_config
from vais_voice.datasets.adapters.fleurs import FleursAdapter
from vais_voice.datasets.adapters.ksc2 import KSC2Adapter

ADAPTERS = {"ksc2": KSC2Adapter, "fleurs": FleursAdapter, "asvspoof2021": ASVspoof2021Adapter}


def adapter_from_config(path: Path) -> DatasetAdapter:
    config = load_source_config(path)
    try:
        adapter = ADAPTERS[config.adapter]
    except KeyError as exc:
        raise ValueError(f"Unknown dataset adapter: {config.adapter}") from exc
    return adapter(config, path)


__all__ = ["DatasetAdapter", "adapter_from_config"]
