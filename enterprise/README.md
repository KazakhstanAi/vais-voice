# VAIS Enterprise MVP

Offline-first evaluation infrastructure for KZ/RU RAG, copilots and AI agents.

## Run locally

```powershell
cd enterprise\backend
..\..\.venv\Scripts\python.exe -m pip install -r requirements.txt
..\..\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8010
```

In a second terminal:

```powershell
cd enterprise\frontend
npm install
npm run dev
```

The frontend defaults to `http://127.0.0.1:8010`. It works with the bundled
Mock A and Mock B targets and never needs an external model or private corpus.

## Reproducibility

The demo dataset is self-written and lives in `benchmark/kzru-enterprise-v0.1`.
Each completed run gets a new UUID and a snapshot under `artifacts/runs/<run_id>`
containing `run.json`, `results.jsonl`, `metrics.json`, `manifest.json`, and
`sha256.txt`. Finished runs are never mutated.

`GenericHTTPJSONTarget` is registered as metadata only in this MVP. Credential
persistence is deliberately unsupported; a secret is accepted for a single
request only and is never saved or logged.

## Test and build

```powershell
cd enterprise\backend
..\..\.venv\Scripts\python.exe -m pytest tests -q

cd ..\frontend
npm run typecheck
npm run build
```

## Deployment shape

Run the API on `127.0.0.1:8010`, then route
`enterprise.vaislabs.com` through the existing named Cloudflare Tunnel. See
`docs/deployment.md`; tunnel credentials and API keys must stay outside git.
