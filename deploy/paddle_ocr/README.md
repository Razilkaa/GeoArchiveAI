# PaddleOCR GPU service

Runs PP-OCRv5 Russian OCR on GPU 1 and exposes a stable internal HTTP contract.

```bash
sudo docker compose up -d --build
curl http://127.0.0.1:18080/health
curl -F file=@page.jpg http://127.0.0.1:18080/v1/ocr
```

The service binds host port `18080`; production ingress and authentication belong
to the orchestration API, not this model container.

`Dockerfile.bootstrap` rebuilds the base image on a clean server. The regular
`Dockerfile` reuses the local `paddle-base` image to avoid downloading the 1.2 GB
Paddle GPU wheel during routine application rebuilds.

On the Windows orchestrator, run `..\start_geoarchive.cmd` to start the SSH
forward, FastAPI control plane, and Streamlit UI. OCR traffic stays on
`http://127.0.0.1:18080` and bypasses environment HTTP proxies.
