from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    project_root: Path = Path(os.getenv("GEOARCHIVE_ROOT", r"C:\FINAM\Conference"))
    ocr_api_url: str = os.getenv("OCR_API_URL", "http://127.0.0.1:18080")
    ragflow_url: str = os.getenv("RAGFLOW_URL", "https://ragflow-dev.finam.ru/api/v1")
    proxy_url: str = os.getenv("GEOARCHIVE_PROXY", "socks5h://127.0.0.1:7777")
    auto_intake_enabled: bool = os.getenv("AUTO_INTAKE_ENABLED", "true").casefold() == "true"
    auto_intake_interval_s: int = int(os.getenv("AUTO_INTAKE_INTERVAL_S", "10"))
    auto_intake_settle_s: int = int(os.getenv("AUTO_INTAKE_SETTLE_S", "30"))
    auto_max_reports: int = int(os.getenv("AUTO_MAX_REPORTS", "2"))

    @property
    def runs_root(self) -> Path:
        return self.project_root / "runs"

    @property
    def inbox_root(self) -> Path:
        return self.project_root / "reports_inbox"

    @property
    def staging_root(self) -> Path:
        return self.project_root / "reports_staging"

    @property
    def pipeline_root(self) -> Path:
        return self.project_root / "pipeline"


settings = Settings()
