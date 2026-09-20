# VAIS-KZRU synthetic speech dataset v0.1

This subsystem builds research data; it is not VAIS VoiceGen and it does not train a detector.
Multiple generator families reduce dependence on artifacts from one architecture. Generators tagged
`unseen_test` are protected from training, while `external_challenge` data is a separate research
partition and must not be merged into ordinary evaluation. High performance on generators seen
during training does not establish general synthetic-speech detection capability. The main research
question is generalization to unseen generators and real-world conditions.

## Text identity and scope

Canonical real rows may carry `transcript`, `normalized_text`, and `text_id`. Inventory extraction
normalizes validated transcripts and derives a deterministic `text_id` from language plus normalized
text. Exact repeated content therefore shares an identity even across source records. Real audio and
all synthetic variants with that identity are connected by the split logic; crossing protected split
boundaries is an error. Synthetic rows retain `source_text_id` rather than overloading
`parent_sample_id`, which continues to describe audio derivation.

Synthetic v0.1 supports only `kk` and `ru`. `kk_ru` generation is deferred until a reviewed
code-switching protocol, inventory, and evaluation design exist.

Build an inventory from an already prepared/split canonical real manifest:

```bash
python -m vais_voice.generation inventory \
  --manifest data/processed/kzru_real_v01/manifest.split.parquet \
  --output data/manifests/local/kzru_real_v01_texts.jsonl
```

## Provenance and licensing

Every generator config separates model provenance, research role, runtime settings, voices, and
generation parameters. Unknown checkpoint revisions or licensing facts remain `null`/TODO. Repository
visibility is never interpreted as commercial permission. A real adapter refuses unresolved
licensing unless the operator explicitly records `license_review_acknowledged: true` after review.
That acknowledgement records a decision; it is not a license grant.

The protocol roles are:

- `train`: eligible for training, validation, and known-generator evaluation.
- `validation`: excluded from training and eligible for validation/test only.
- `unseen_test`: protected evaluation only (MMS and OmniVoice in v0.1).
- `external_challenge`: separate challenge data only (AIT-Syn in v0.1).

## Plan, run, resume, report

Planning does no synthesis and imports no model runtime:

```bash
python -m vais_voice.generation plan \
  --config configs/generation/kzru_synthetic_v01_pilot.yaml
python -m vais_voice.generation run \
  --plan data/processed/generation-plans/kzru_synthetic_v01_pilot/generation_jobs.jsonl
python -m vais_voice.generation report \
  --run data/processed/generation-plans/kzru_synthetic_v01_pilot/run
```

Job IDs hash immutable text/model/voice/parameter/seed inputs. `skip` accepts an existing output only
when its recorded checksum and audio structure are valid; `fail` rejects every collision;
`regenerate` explicitly replaces a job output. Results are persisted after every job. The run emits
canonical JSONL/Parquet synthetic manifests, a runtime snapshot, and a machine-readable report based
only on completed outputs. Generator output rate and format are preserved and recorded; detector
resampling belongs to later preprocessing.

On Windows, the Python Piper runtime receives text through a temporary UTF-8 `--input-file` rather
than stdin. A UTF-8 pipe can otherwise be decoded by the child Python process using the active
Windows console code page, producing valid but linguistically corrupted audio. The temporary input
is removed immediately after inference.

The pilot limits of 250 Kazakh and 250 Russian texts are YAML values. Nothing downloads models or
generates audio during package installation, imports, tests, or CI. The deterministic sine adapter is
not speech and exists only to test infrastructure offline.

## Adding a generator

1. Add a strict generator YAML with reviewed identity, revision, role, language, and license state.
2. Implement `GeneratorAdapter.validate_runtime()` and `synthesize()` without import-time downloads.
3. Register it lazily in `generation/registry.py`; represent unsupported seed/voice capabilities
   explicitly.
4. Test the boundary with fixtures, then record actual local verification status honestly.
5. Install heavyweight or CUDA-sensitive dependencies in a model-specific reviewed environment.

Silero, VoxCPM, Qwen3-TTS, MMS, OmniVoice, and AIT-Syn currently have explicit runtime-unverified
boundaries. Piper has a local CLI integration. The pinned Windows pilot configs for
`kk_KZ-issai-high` and `ru_RU-dmitri-medium` were verified with `piper-tts==1.8.0` in a dedicated
environment; their runtime and model paths point into the ignored `runs/piper-real-v01/` directory
and are guarded by SHA-256. Re-download those exact pinned artifacts before reproducing on a new
checkout. Piper is exposed through the `generation-piper` optional dependency and is intentionally
not a base dependency.

The current Piper runtime package is GPL-3.0-or-later, while the pinned voice repository revision is
MIT. The selected model cards identify CC BY 4.0 Kazakh source data and CC0 Russian source data, but
do not state an unambiguous license for the resulting checkpoints. The configs therefore keep the
checkpoint license explicitly unknown and retain `commercial_use_status: needs_review`; this pilot
is a research generation run, not a commercial-clearance decision.
