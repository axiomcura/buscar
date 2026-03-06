"""Pytest configuration for local src-layout imports."""

import sys
from pathlib import Path

# Add src directory to Python path for local test execution.
project_root = Path(__file__).parent.parent
src_dir = project_root / "src"
sys.path.insert(0, str(src_dir))
