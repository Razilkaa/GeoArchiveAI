import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.routers import maps, reports, system
from app.config import settings
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
app.include_router(system.router)
app.include_router(reports.router)
app.include_router(maps.router)
