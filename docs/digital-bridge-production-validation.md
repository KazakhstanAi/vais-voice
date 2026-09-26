# Digital Bridge detector production validation

Validation date: 2026-09-26. Scope: controlled in-process FastAPI boundary on the exhibition
workstation, CUDA, no bearer token, no tunnel, and non-personal benchmark audio. This validates the
same backend used by gateway `/api/*`; it does not prove external tunnel uptime.

`/health` returned `status=ok`, `mockMode=false`, `modelLoaded=true`, CUDA device, model
`vais-detector-logmel-cnn-kk-v0.1`, threshold 0.99609375, and checkpoint SHA-256
`3a711c2fabfc85c7239940a229ed77adff1b28077760faa6b1dd80e9ed942ce0`. `/model` returned the same
identity and hash.

Ten sequential valid `/detect` calls succeeded (10/10). Latency was 22.743–344.440 ms, mean 58.715
ms, including first-call warm-up. Every response used the same model and checkpoint. Torch reported
9,899,520 allocated CUDA bytes and 27,262,976 reserved bytes after the run.

Input checks:

- empty file: HTTP 400;
- unsupported extension: HTTP 400;
- oversized file (>25 MiB): HTTP 413;
- malformed WAV: HTTP 400;
- 0.1-second WAV: HTTP 200 with analyzed duration 0.1 s;
- 5.027-second WAV: HTTP 200 and `analyzedSeconds=4.0`, matching checkpoint maximum.

Blocker: the malformed-WAV detail includes the absolute temporary file path emitted by libsndfile.
This violates the no-local-path error requirement. The gateway/service owner must replace backend
exception text with a fixed public message while retaining the detailed exception only in protected
server logs, then repeat this test. Backend timeout behavior and external `/api/*` tunnel routing
were not evaluated by the in-process client. No production secret or deployment was changed.

Reproduce with `scripts/validate_local_detector_api.py`; output is printed to stdout and contains no
token or raw audio.
