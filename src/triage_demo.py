"""Compatibility entry point for the Reserve Watch dashboard."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.triage_demo import main

if __name__ == "__main__":
    main(legacy_ui=True)
