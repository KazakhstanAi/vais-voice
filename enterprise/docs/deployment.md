# Cloudflare Tunnel route

The API intentionally binds only to localhost:

```powershell
cd enterprise\backend
..\..\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8010
```

In the existing named tunnel configuration, add this ingress rule before the
terminal catch-all rule:

```yaml
- hostname: enterprise.vaislabs.com
  service: http://127.0.0.1:8010
```

Register DNS once with the existing named tunnel:

```powershell
cloudflared tunnel route dns d77f9bdf-dc1e-43e9-87db-2a6f89f9848c enterprise.vaislabs.com
```

Do not commit `cert.pem`, tunnel JSON credentials, tokens, or API keys. Restart
the tunnel only in a maintenance window because it also routes the existing
voice inference hostname.
