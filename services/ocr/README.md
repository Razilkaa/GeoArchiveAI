# PaddleOCR API

GPU-сервис принимает изображение и возвращает текст, confidence и координаты
OCR-боксов. Основной стенд обращается к нему по `OCR_API_URL`, по умолчанию
`http://127.0.0.1:18080`.

```powershell
docker compose up -d --build
```

Приложение: `app.py`. Docker-конфигурация: `compose.yaml`.
