# Web UI

Production UI is a React/TypeScript SPA. It reads reports, statuses, previews,
downloads and RAG answers only through FastAPI.

```powershell
npm.cmd install
$env:GEOARCHIVE_API_URL = "http://127.0.0.1:8765"
npm.cmd run dev
```

The Vite development server proxies `/api` to FastAPI. `npm.cmd run build`
creates the production bundle used by `deploy/Dockerfile.web`.

`app.py` remains a compatibility Streamlit client. It also uses HTTP artifact
URLs and does not read `runs/`, source reports or `results/` directly.
