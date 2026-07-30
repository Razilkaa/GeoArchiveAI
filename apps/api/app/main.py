import asyncio
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings

project_root = str(settings.project_root)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app.routers import maps, reports, system
from app.services.automation import InboxMonitor, stop_monitor


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = None
    if settings.auto_intake_enabled:
        task = asyncio.create_task(InboxMonitor().run(), name="geoarchive-inbox-monitor")
    app.state.inbox_monitor = task
    yield
    if task is not None:
        await stop_monitor(task)


app = FastAPI(
    title="GeoArchiveAI API",
    description="Control plane for report intake, OCR, RAGFlow and map digitization.",
    version="0.4.0",
    lifespan=lifespan,
)
if settings.cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
app.include_router(system.router)
app.include_router(reports.router)
app.include_router(maps.router)
