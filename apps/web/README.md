# Streamlit web client

Тонкий интерфейс GeoArchiveAI. Список отчётов, загрузка, запуск, статусы и
результаты приходят только через FastAPI на `http://127.0.0.1:8765`.

```powershell
cd C:\FINAM\Conference\apps\web
python -m streamlit run app.py --server.port 8501
```

Основной способ запуска всего стенда: `scripts\start_geoarchive.cmd`.
