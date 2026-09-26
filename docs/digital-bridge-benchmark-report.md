# VAIS Voice KZ/RU Benchmark Pilot v0.2

## 1. Executive summary

The release proves that VAIS can pin a voice model and runtime, retain exact audio and hashes, and
connect detector results to traceable checkpoint and dataset artifacts. The original training
worktree was dirty, so the training run is not claimed as fully reproducible from its commit. It does not prove general
deepfake detection. Protected unseen-generator evaluation remains planned.

## 2–6. Scope, models, dataset, review, and dimensions

The public paired diagnostic compares Piper standalone 1.2.0 with piper-tts 1.8.0 using the same
Kazakh model, voice, prompt, and one sample per runtime. Exact source and public hashes match.
Review findings exist, but reviewer identity/count and structured dimension ratings do not.
Naturalness interpretations are marked derived. Synthetic detectability and the other seven
dimensions are not evaluated at paired-sample level; detector diagnostics remain separate run-level
evidence. Russian voice evidence may be displayed elsewhere but is not ranked
against Kazakh evidence and is not part of the detector result.

The detector dataset contains 1,500 Kazakh items: train 1,064 (707 real, 357 synthetic), validation
203 (136 real, 67 synthetic), and test 233 (157 real, 76 synthetic). Real speech is FLEURS; synthetic
speech is one legacy Piper 1.2 generator. Exact hash/text/source overlap across splits is zero.

## 7–10. Detector and diagnostics

Checkpoint `vais-detector-logmel-cnn-kk-v0.1` separates Kazakh FLEURS speech from one seen Piper
generator on the pilot split. At validation-EER threshold 0.99609375, test confusion is TN 157, FP
0, FN 2, TP 74. Threshold-free seen metrics are ROC-AUC 1.0, EER 0.0, and FPR@TPR95 0.0. The score
is an uncalibrated model evidence score, not a probability.

The unseen slice has 157 real and zero synthetic samples, so ROC-AUC, EER, and FPR@TPR95 are null.
No protected metric is published. Duration is a material confound: test median is 11.58 s for real
versus 6.7248 s for synthetic. Codec and prepared sample rate are matched; speaker independence,
replay robustness, compression robustness, and channel generalization are unproven.

## 11. Provenance

Checkpoint SHA-256 is
`3a711c2fabfc85c7239940a229ed77adff1b28077760faa6b1dd80e9ed942ce0`. Dataset manifest SHA-256
is `f74ff1261c418c49acd923b8c89b300b4bf3a6c94056990bc630590ccf2f3afe` and preparation SHA-256
is `81425d2aea89d787350f88b2f20da6fd666cf2d638e1e34a4181486a5a2ee66e`. Training recorded commit
`26ae06bfb0fffde2775bce9200c3ab75b9dc8bda` with a dirty worktree; that limitation is retained.

## 12. Limitations

One language and one seen generator are covered. FLEURS speaker IDs are unavailable. Human-review
metadata is incomplete. Runtime-pair sample count is one. License review for model/runtime bundle
redistribution is not equivalent to dataset licensing. The local GPU service and tunnel depend on
workstation uptime. No calibrated probability, general authorship claim, or production antifraud
claim is supported.

## 13–14. Reproduction and next milestone

Validate the immutable snapshot with `python -m vais_voice.benchmark.validate
artifacts/benchmark/kzru-voice-benchmark-pilot-v0.2.json`. Run `pytest` and `ruff check .`.
Reproduction of model inference additionally requires the ignored checkpoint and dataset artifacts
documented in the checkpoint card. The next milestone is a legally reviewed, quality-gated,
independent Kazakh generator corpus followed by a frozen, one-shot protected evaluation.
