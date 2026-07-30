from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from geoarchive.settings import Settings, load_settings


settings: Settings = load_settings()

__all__ = ["Settings", "settings"]
