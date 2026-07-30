# Server deployment

The production stack contains four services:

```text
nginx/react -> FastAPI -> shared file queue -> report worker
                         |                 -> RAGFlow
                         +-> GPU PaddleOCR
```

1. Copy `.env.example` to `.env` and set `RAGFLOW_URL`.
2. Put `ragflow_token.txt` and `tokent.txt` in `secrets/`.
3. Place the SRR shapefile set in `shapes/`.
4. Create or select the persistent host directory configured by
   `GEOARCHIVE_DATA_PATH`.
5. Run:

```powershell
docker compose --env-file .env -f deploy/compose.yaml up -d --build
```

The web application is published on `GEOARCHIVE_HTTP_PORT` (8080 by default).
Runtime state and user results live under the mounted `/data`; they are not
baked into images. API and worker share the same image and configuration, but
run as separate processes. The worker claims atomic JSON requests from
`/data/runs/_queue`.

For `msa-aiad01-ap04`, install Docker Engine, Compose v2 and NVIDIA Container
Toolkit. The production OCR image is self-contained and does not depend on the
historical local `geoarchive/paddle-ocr:paddle-base` image.
