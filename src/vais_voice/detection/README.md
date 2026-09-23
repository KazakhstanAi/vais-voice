# Detection

`base.py` defines `Detector.predict_score(audio_path: Path) -> float`. `DummyDetector` remains an
explicit infrastructure-only constant predictor.

Detector Baseline v0.1 implements:

- `logmel_cnn`: log-Mel frontend and compact CNN (Baseline A);
- `wav2vec2_frozen_mlp`: frozen official torchaudio Wav2Vec2 encoder and trainable MLP head
  (Baseline B);
- manifest-backed, identical train/inference waveform loading;
- deterministic training and early stopping;
- checkpoint selection by validation ROC-AUC, then validation loss;
- EER threshold selection on validation only;
- protected test export with KK/RU, per-generator, seen, and unseen slices;
- checkpoint hashes and runtime provenance;
- CLI and FastAPI-compatible checkpoint inference.

The fixture test trains on mathematical waveforms only and proves software behavior, not detection
quality. A checkpoint becomes a research result only after training on the reviewed v0.1 corpus.
