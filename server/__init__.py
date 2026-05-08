"""FastAPI server package for the VR interview system."""

from pathlib import Path
import sys

SRC_ROOT = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_ROOT))
