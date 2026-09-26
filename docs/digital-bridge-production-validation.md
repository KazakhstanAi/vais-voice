# Digital Bridge detector production validation

Validation date: 2026-09-26. Scope: controlled in-process FastAPI boundary on the exhibition
workstation, CUDA, no bearer token, no tunnel, and non-personal benchmark audio. This validates the
same backend used by gateway `/api/*`; it does not prove external tunnel uptime.

`/health` returned `status=ok`, `mockMode=false`, `modelLoaded=true`, CUDA device, model
`vais-detector-logmel-cnn-kk-v0.1`, threshold 0.99609375, and checkpoint SHA-256
`3a711c2fabfc85c7239940a229ed77adff1b28077760faa6b1dd80e9ed942ce0`. `/model` returned the same
identity and hash.

Ten sequential valid `/detect` calls succeeded (10/10). Post-fix latency was 24.903–416.025 ms, mean 66.207
ms, including first-call warm-up. Every response used the same model and checkpoint. Torch reported
9,899,520 allocated CUDA bytes and 27,262,976 reserved bytes after the run.

Input checks:

- empty file: HTTP 400;
- unsupported extension: HTTP 400;
- oversized file (>25 MiB): HTTP 413;
- malformed WAV and FLAC: HTTP 400 with the fixed public decode message;
- audio MIME with non-audio content: HTTP 400 with the same fixed message;
- unsupported encoding: HTTP 400 with the same fixed message;
- 0.1-second WAV: HTTP 200 with analyzed duration 0.1 s;
- 5.027-second WAV: HTTP 200 and `analyzedSeconds=4.0`, matching checkpoint maximum.

The path-disclosure blocker is resolved. Public decode failures return only `Audio could not be
decoded or contains an unsupported encoding.` The server log retains the original exception,
generated/requested UUID, and `temporary_decoding` stage; it does not log the bearer token or raw
audio. Backend timeout behavior and external `/api/*` tunnel routing were not evaluated by the
in-process client at this stage.

Reproduce with `scripts/validate_local_detector_api.py`; output is printed to stdout and contains no
token or raw audio.
