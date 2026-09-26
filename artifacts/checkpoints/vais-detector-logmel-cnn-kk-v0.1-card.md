# vais-detector-logmel-cnn-kk-v0.1 checkpoint card

Compact log-Mel CNN trained on `kk-detector-pilot-v0.1`, split
`kk-known-generator-v0.1`. The checkpoint was selected by validation ROC-AUC then validation loss;
threshold 0.99609375 was selected at validation EER. Neither selection used test data.

Checkpoint SHA-256: `3a711c2fabfc85c7239940a229ed77adff1b28077760faa6b1dd80e9ed942ce0`.
Manifest SHA-256: `f74ff1261c418c49acd923b8c89b300b4bf3a6c94056990bc630590ccf2f3afe`.
Preparation SHA-256: `81425d2aea89d787350f88b2f20da6fd666cf2d638e1e34a4181486a5a2ee66e`.

Input is mono-normalized 16 kHz audio, truncated/padded to four seconds. Output is an uncalibrated
synthetic evidence score; it is not an authorship probability. The only valid published result is
the seen-generator Kazakh diagnostic. See the JSON sidecar and leakage audit for full limitations.
