from fastapi import FastAPI

from app.routers import reports, system


app = FastAPI(
    title="GeoArchiveAI API",
    description="Control plane for report intake, OCR, RAGFlow and extraction agents.",
    version="0.2.0",
)
app.include_router(system.router)
app.include_router(reports.router)
