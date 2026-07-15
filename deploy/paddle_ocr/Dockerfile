FROM geoarchive/paddle-ocr:paddle-base

ENV PYTHONUNBUFFERED=1 \
    PADDLE_OCR_BASE_DIR=/models/paddleocr \
    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
    PADDLE_PDX_MODEL_SOURCE=BOS

RUN python -m pip install --no-cache-dir \
    paddlex==3.1.0 \
    paddleocr==3.1.0 \
    fastapi==0.115.14 \
    "uvicorn[standard]==0.34.3" \
    python-multipart==0.0.20

WORKDIR /app
COPY app.py /app/app.py

EXPOSE 8080
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
