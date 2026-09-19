# Manifest contract

JSONL / Parquet; implementation: src/vais_voice/datasets/schema.py.
Paths are relative to raw root before preparation, manifest directory afterwards.

| Field | Contract |
| --- | --- |
| sample_id | Unique alphanumeric/underscore/hyphen ID |
| path | Relative audio path, no traversal |
| label | real / synthetic |
| language | kk / ru / kk_ru / other |
| speaker_id | Canonical speaker identity |
| related_speaker_ids | Optional involved/reference speakers |
| source_dataset | Dataset name/version |
| source_id | Canonical source recording; shared by derivatives |
| generator_id | Required synthetic, null real |
| codec | Source-channel label, unknown allowed |
| sample_rate, duration_sec | Positive; measured in prepare |
| split | train / val / test or null before splitting |
| license | Reviewed license/terms reference |
| original_path | Original provenance path |
| parent_sample_id | Parent in same manifest, or null |
| rights_reference | Consent/license evidence reference |
| usage_permission | Must be approved |
| condition | clean/noise/etc., defaults clean |
| source_sha256, processed_sha256 | Computed integrity hashes |
| attack_id | Optional ASVspoof attack/system identifier |
| source_record_id | Optional verbatim source-side record identifier |
| missing_metadata | Explicit names of unavailable metadata fields |

No derivatives or speakers cross splits. Explicit unseen generators never enter train/val.
Approval metadata does not replace rights review. Private manifests belong in ignored local/.
