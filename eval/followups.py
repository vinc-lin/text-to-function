from __future__ import annotations
import json
from pathlib import Path

def load_followups(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
