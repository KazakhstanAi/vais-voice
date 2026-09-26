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

## Remote release verification

After deployment, remote D1 seeding, backend-token rotation, and FastAPI restart, the public gateway
reported `inference=configured`, `mockMode=false`, the expected model/checkpoint, and CUDA. Public
malformed WAV, malformed FLAC, audio-MIME/non-audio content, and unsupported-encoding requests all
returned HTTP 400 with the fixed decode message and no machine path. A zero-byte upload returned
HTTP 400 `Empty audio`. Automated log capture confirmed the original exception, request UUID, and
`temporary_decoding` stage while excluding raw audio and bearer tokens.

Remote `/api/benchmark` returned release `kzru-voice-benchmark-pilot-v0.2` and snapshot SHA-256
`93877ceb35abf07d86b5f0ca8ff516783cf9e21393c58e5ba680e2b58b441a4c`. Downloaded public hashes
were the same snapshot hash, Piper 1.2 WAV
`5c0469c08ec66969a7e758367af7d123b1c856eb0fb0c20fc8c2925b81e53048`, and Piper 1.8 WAV
`22e237710069cd6786ba49044d1fd76dd71378ca6d395c2805875da1c150420f`.

Reproduce with `scripts/validate_local_detector_api.py`; output is printed to stdout and contains no
token or raw audio.
