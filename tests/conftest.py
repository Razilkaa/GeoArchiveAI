from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
API_ROOT = ROOT / "apps" / "api"
for path in (ROOT, API_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
