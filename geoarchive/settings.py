from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _path(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, str(default))).expanduser().resolve()


def _bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().casefold() in {"1", "true", "yes", "on"}


def _optional_url(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name, default or "").strip()
    return None if value.casefold() in {"", "none", "off"} else value.rstrip("/")


@dataclass(frozen=True)
class Settings:
    project_root: Path
    data_root: Path
    runs_root: Path
    inbox_root: Path
    staging_root: Path
    results_root: Path
    pipeline_root: Path
    ragflow_token_path: Path
    llm_credentials_path: Path
    ocr_api_url: str
    ragflow_url: str
    ragflow_dataset_id: str | None
    proxy_url: str | None
    ragflow_verify_tls: bool
    ragflow_retrieval_timeout_s: float
    ragflow_retrieval_attempts: int
    ragflow_retry_backoff_s: float
    ragflow_page_size: int
    ragflow_similarity_threshold: float
    ragflow_vector_similarity_weight: float
    ragflow_top_k: int
    llm_model: str
    enable_enrichment: bool
    auto_intake_enabled: bool
    auto_intake_interval_s: int
    auto_intake_settle_s: int
    auto_max_reports: int
    map_max_jobs: int
    max_upload_bytes: int
    s3_endpoint_url: str | None
    s3_bucket: str | None
    s3_access_key: str | None
    s3_secret_key: str | None
    s3_region: str
    s3_source_prefix: str
    s3_results_prefix: str
    job_execution_mode: str
    queue_poll_interval_s: float
    cors_origins: tuple[str, ...]

    @classmethod
    def from_environment(cls) -> "Settings":
        project_root = _path("GEOARCHIVE_ROOT", PROJECT_ROOT)
        data_root = _path("GEOARCHIVE_DATA_ROOT", project_root)
        secrets_root = _path("GEOARCHIVE_SECRETS_ROOT", project_root / "secrets")
        cors = tuple(
            item.strip()
            for item in os.environ.get("GEOARCHIVE_CORS_ORIGINS", "").split(",")
            if item.strip()
        )
        execution_mode = os.environ.get("JOB_EXECUTION_MODE", "subprocess").strip().casefold()
        if execution_mode not in {"subprocess", "queue"}:
            raise ValueError("JOB_EXECUTION_MODE must be 'subprocess' or 'queue'")
        return cls(
            project_root=project_root,
            data_root=data_root,
            runs_root=_path("GEOARCHIVE_RUNS_ROOT", data_root / "runs"),
            inbox_root=_path("GEOARCHIVE_INBOX_ROOT", data_root / "reports_inbox"),
            staging_root=_path("GEOARCHIVE_STAGING_ROOT", data_root / "reports_staging"),
            results_root=_path("GEOARCHIVE_RESULTS_ROOT", data_root / "results"),
            pipeline_root=_path("GEOARCHIVE_PIPELINE_ROOT", project_root / "pipeline"),
            ragflow_token_path=_path(
                "RAGFLOW_TOKEN_FILE", secrets_root / "ragflow_token.txt"
            ),
            llm_credentials_path=_path(
                "LLM_CREDENTIALS_FILE", secrets_root / "tokent.txt"
            ),
            ocr_api_url=(_optional_url("OCR_API_URL", "http://127.0.0.1:18080") or ""),
            ragflow_url=(_optional_url("RAGFLOW_URL", "https://ragflow-dev.finam.ru/api/v1") or ""),
            ragflow_dataset_id=os.environ.get("RAGFLOW_DATASET_ID", "").strip() or None,
            proxy_url=_optional_url("RAGFLOW_PROXY", _optional_url("GEOARCHIVE_PROXY")),
            ragflow_verify_tls=_bool("RAGFLOW_VERIFY_TLS", True),
            ragflow_retrieval_timeout_s=float(
                os.environ.get("RAGFLOW_RETRIEVAL_TIMEOUT_S", "30")
            ),
            ragflow_retrieval_attempts=max(
                1, int(os.environ.get("RAGFLOW_RETRIEVAL_ATTEMPTS", "2"))
            ),
            ragflow_retry_backoff_s=max(
                0.0, float(os.environ.get("RAGFLOW_RETRY_BACKOFF_S", "1"))
            ),
            ragflow_page_size=int(os.environ.get("RAGFLOW_PAGE_SIZE", "5")),
            ragflow_similarity_threshold=float(
                os.environ.get("RAGFLOW_SIMILARITY_THRESHOLD", "0.1")
            ),
            ragflow_vector_similarity_weight=float(
                os.environ.get("RAGFLOW_VECTOR_SIMILARITY_WEIGHT", "0.3")
            ),
            ragflow_top_k=int(os.environ.get("RAGFLOW_TOP_K", "64")),
            llm_model=os.environ.get("GEOARCHIVE_LLM_MODEL", "openai/gpt-4o-mini"),
            enable_enrichment=_bool("ENABLE_ENRICHMENT", False),
            auto_intake_enabled=_bool("AUTO_INTAKE_ENABLED", True),
            auto_intake_interval_s=int(os.environ.get("AUTO_INTAKE_INTERVAL_S", "10")),
            auto_intake_settle_s=int(os.environ.get("AUTO_INTAKE_SETTLE_S", "30")),
            auto_max_reports=int(os.environ.get("AUTO_MAX_REPORTS", "2")),
            map_max_jobs=int(os.environ.get("MAP_MAX_JOBS", "2")),
            max_upload_bytes=int(os.environ.get("MAX_UPLOAD_BYTES", str(30 * 1024**3))),
            s3_endpoint_url=_optional_url("S3_ENDPOINT_URL"),
            s3_bucket=os.environ.get("S3_BUCKET", "").strip() or None,
            s3_access_key=os.environ.get("S3_ACCESS_KEY", "").strip() or None,
            s3_secret_key=os.environ.get("S3_SECRET_KEY", "").strip() or None,
            s3_region=os.environ.get("S3_REGION", "us-east-1").strip(),
            s3_source_prefix=os.environ.get("S3_SOURCE_PREFIX", "source").strip("/"),
            s3_results_prefix=os.environ.get("S3_RESULTS_PREFIX", "results").strip("/"),
            job_execution_mode=execution_mode,
            queue_poll_interval_s=float(os.environ.get("QUEUE_POLL_INTERVAL_S", "2")),
            cors_origins=cors,
        )

    @property
    def queue_root(self) -> Path:
        return self.runs_root / "_queue"


@lru_cache(maxsize=1)
def load_settings() -> Settings:
    return Settings.from_environment()
