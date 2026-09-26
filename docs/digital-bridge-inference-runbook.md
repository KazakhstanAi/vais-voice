# Digital Bridge detector inference runbook

The service is workstation-hosted and depends on the local NVIDIA GPU, power state, FastAPI
process, and tunnel. Sleep or shutdown makes it offline; treat this as an exhibition risk.

1. In `vais-voice`, use `.venv\Scripts\python.exe` and verify CUDA with
   `python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name())"`.
2. Verify `runs/detector-logmel-cnn-kk-v0.1/best.pt` SHA-256 equals
   `3a711c2fabfc85c7239940a229ed77adff1b28077760faa6b1dd80e9ed942ce0`. Stop on mismatch.
3. In `vais-voice-web`, install the ML service requirements if needed. Set
   `VAIS_ML_MOCK_MODE=false`, `VAIS_ML_VOICE_REPO` to the local research repo,
   `VAIS_ML_DETECTOR_CHECKPOINT` to the checkpoint, and `VAIS_ML_DETECTOR_DEVICE=cuda`. Supply the
   API token only through the environment/secret store; never write it to the repository.
4. Start from `services/ml-api` with the research Python:
   `python -m uvicorn app.main:app --host 127.0.0.1 --port 8000`.
5. Check `/health`: `mockMode` must be false, `modelLoaded` true, device `cuda`, and model/hash must
   match this document. A failed load is degraded and must never fall back to mock automatically.
6. Warm up with one controlled non-personal WAV, then validate `/model` and `/detect`. Confirm the
   response exposes the same model ID, checkpoint hash, threshold, and `mock: false`.
7. Start the preconfigured named Cloudflare Tunnel. Do not commit its credential, token, or temporary
   hostname. Update the Worker backend secret through the approved secret command, then run gateway
   health and one controlled upload. Do not log authorization headers or raw personal audio.
8. For shutdown, stop the tunnel first and then FastAPI. For recovery, verify power/network/GPU,
   repeat checkpoint verification, restart FastAPI, warm up, start the tunnel, and repeat smoke.

If the local service is unavailable, the frontend must show offline/unavailable. Mock mode is for
explicit development only and cannot masquerade as detector inference. TTS remains a documented
501 when real generation is not connected.
