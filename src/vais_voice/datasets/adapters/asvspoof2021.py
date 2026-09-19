"""Local ASVspoof 2021 protocol adapter (bonafide -> real, spoof -> synthetic)."""

from vais_voice.datasets.adapters.base import (
    AdaptedSample,
    DatasetAdapter,
    deterministic_id,
    inspect_audio,
)
from vais_voice.datasets.schema import Sample
from vais_voice.preprocessing.audio import SUPPORTED


class ASVspoof2021Adapter(DatasetAdapter):
    def validate(self) -> None:
        super().validate()
        if self.config.metadata.intended_role != "external_sanity":
            raise ValueError("ASVspoof 2021 must use intended_role: external_sanity")

    def iter_samples(self) -> list[AdaptedSample]:
        self.validate()
        if not self.config.protocol_file:
            raise ValueError("ASVspoof 2021 requires protocol_file")
        protocol = self.source_file(self.config.protocol_file)
        if not protocol.is_file():
            raise ValueError(f"ASVspoof protocol not found: {protocol}")
        audio_root = self.source_file(self.config.audio_dir)
        if not audio_root.is_dir():
            raise ValueError(f"ASVspoof audio directory not found: {audio_root}")
        by_stem = {
            p.stem: p for p in sorted(audio_root.rglob("*")) if p.suffix.lower() in SUPPORTED
        }
        result = []
        for number, line in enumerate(protocol.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 6 and parts[5] in {"bonafide", "spoof"}:
                # Official 2021 keys: speaker trial codec source attack label ...
                speaker, utterance, _, _, attack, protocol_label = parts[:6]
            elif len(parts) >= 5:
                # Compact five-column exports used by some local tools.
                speaker, utterance, _, attack, protocol_label = parts[:5]
            else:
                raise ValueError(f"Invalid ASVspoof protocol row {number}: expected 5+ fields")
            if protocol_label not in {"bonafide", "spoof"}:
                raise ValueError(f"Unknown ASVspoof label {protocol_label!r} in row {number}")
            source = by_stem.get(utterance)
            if source is None:
                raise ValueError(f"ASVspoof audio not found for {utterance!r}")
            label = "real" if protocol_label == "bonafide" else "synthetic"
            attack_id = None if attack == "-" else attack
            generator_id = None if label == "real" else (attack_id or "asvspoof_unknown_attack")
            relative = source.relative_to(self.root).as_posix()
            result.append(
                AdaptedSample(
                    Sample(
                        sample_id=deterministic_id(self.config.metadata.source_id, utterance),
                        path=relative,
                        label=label,
                        language=self.config.language or "other",
                        speaker_id=speaker,
                        source_dataset=self.config.metadata.source_id,
                        source_id=utterance,
                        generator_id=generator_id,
                        attack_id=attack_id,
                        license=self.config.metadata.license or "",
                        original_path=relative,
                        rights_reference=self.config.metadata.license_url
                        or self.config.metadata.homepage_url
                        or "configured source metadata",
                        usage_permission="approved",
                        source_record_id=utterance,
                        **inspect_audio(source),
                    ),
                    source,
                )
            )
        return result
