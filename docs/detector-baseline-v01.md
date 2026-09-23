# Detector Baseline v0.1 runbook

The baseline is a research pipeline, not a production antifraud model. It trains either a compact
log-Mel CNN or a frozen Wav2Vec2 encoder with an MLP head. Checkpoint selection and the decision
threshold use validation data only; test metrics are computed after the checkpoint is fixed.

## Local corpus build

The pinned FLEURS extraction configs build 1,000 Kazakh and 1,000 Russian real clips. Public FLEURS
metadata does not expose stable speaker identifiers, so this corpus can enforce source, text,
parent, and hash isolation but cannot prove zero speaker overlap. Reports must retain that
limitation.

The current approved synthetic plan contains at most 500 Kazakh Piper 1.2 jobs and 500 Russian
Qwen3-TTS jobs, limited to texts of 141 characters or fewer:

```powershell
.venv\Scripts\python.exe -m vais_voice.generation plan `
  --config configs/generation/kzru_detector_corpus_v01.yaml
.venv\Scripts\python.exe -m vais_voice.generation run `
  --plan runs/detector-corpus-v01/generation/generation_jobs.jsonl
.venv\Scripts\python.exe -m vais_voice.generation report `
  --run runs/detector-corpus-v01/generation/run
```

Qwen runs as a persistent subprocess, so the 1.7B model is loaded once. FLEURS text normalization
capitalizes the initial letter and adds terminal punctuation without changing lexical content. A
text-length-aware token limit rejects truncated/runaway output. Deterministic retry seeds and the
effective accepted seed are recorded in output provenance. Generated audio is neither normalized
nor resampled.

After the generation report and human audit are complete, stage only pass-gated eligible samples:

```powershell
.venv\Scripts\python.exe -m vais_voice.data.compose `
  --real-manifest data/processed/kzru_real_detector_v01/manifest.parquet `
  --synthetic-manifest runs/detector-corpus-v01/generation/run/manifest.synthetic.parquet `
  --output data/staging/kzru_detector_v01
```

The composer requires `quality_gate=pass`, `training_eligible=true`, `diagnostic_only=false`, and
`intended_role=train`. It verifies checksums and stages audio with hard links when possible. Detector
preparation may then decode, downmix, and resample to 16 kHz; the original generator output remains
unchanged.

## Training and inference

Update the detector configs to point at the prepared split corpus, then run:

```powershell
.venv\Scripts\python.exe -m vais_voice.train --config configs/detection/baseline.yaml --dry-run
.venv\Scripts\python.exe -m vais_voice.train --config configs/detection/baseline.yaml
.venv\Scripts\python.exe -m vais_voice.train --config configs/detection/baseline_ssl.yaml
```

Each run writes the best checkpoint, history, validation-selected threshold, predictions, model
metadata, and benchmark JSON with ROC-AUC, EER, and FPR@TPR95 slices. A protected unseen-generator
result must remain marked as planned until a genuinely held-out, quality-approved generator is
available; it must not be simulated from the same generator family.
