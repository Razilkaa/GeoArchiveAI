"""Compatibility entry point. Prefer `uvicorn app.main:app`."""

from app.main import app


__all__ = ["app"]
