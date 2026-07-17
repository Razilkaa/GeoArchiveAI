# Web UI

`app.py` — тонкий Streamlit-клиент FastAPI. Он не читает `runs/` напрямую и
не запускает OCR или оцифровку самостоятельно.

```powershell
python -m streamlit run app.py --server.port 8501
```
