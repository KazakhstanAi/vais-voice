# Detection

base.py defines Detector.predict_score(audio_path: Path) -> float.
DummyDetector always returns 0.5: infrastructure only, not a learned detector.
Predict verifies audio hashes and records dummy provenance.
Real inference and training belong to Phase 1. No weights are downloaded.
