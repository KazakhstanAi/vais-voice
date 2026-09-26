# Leakage audit: vais-detector-logmel-cnn-kk-v0.1

Source: `data/processed/kk_detector_pilot_v01/manifest.split.parquet` (SHA-256
`f74ff1261c418c49acd923b8c89b300b4bf3a6c94056990bc630590ccf2f3afe`). The complete
machine-readable result is `artifacts/audits/detector-logmel-cnn-kk-v0.1-leakage-audit.json`.

All train/validation/test pairings have zero overlap by processed SHA-256, source SHA-256, exact
normalized text, text ID, source ID, source-record ID, and non-null parent sample ID. There are no
duplicate processed hashes in the 1,500-row manifest. All audio is mono 16 kHz WAV after
preparation, so sample rate and codec do not distinguish classes in this prepared corpus.

This does not make the result generator-independent. The same Piper generator family/version is
present in train, validation, and seen test by design. No unseen synthetic generator is present.
FLEURS lacks reliable speaker IDs, so speaker independence cannot be established. Parent IDs are
largely absent, limiting transformed-source checks.

The strongest observed shortcut risk is duration. In test, real median duration is 11.58 seconds
and synthetic median is 6.7248 seconds (means 12.5690 and 6.6734). Train and validation show the
same separation. The model consumes only the first four seconds, which reduces but does not remove
duration, silence, channel, synthesis, or preprocessing fingerprints. Near-duplicate acoustic
content, replay effects, and perceptual similarity were not exhaustively measured.

Decision: existing closed-set evaluation remains publishable only as a pipeline diagnostic. New
training or a protected claim is blocked until an independent, quality-gated generator corpus and
compatible real references pass the same audit.
