"""Filesystem layout. All large artefacts live under the gitignored ``data/``."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
DOCS = ROOT / "docs"
RESULTS = DOCS / "results"


def ensure_dirs() -> None:
    for p in (RAW, PROCESSED, DOCS, RESULTS):
        p.mkdir(parents=True, exist_ok=True)
