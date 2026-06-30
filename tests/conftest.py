"""Shared pytest configuration for the Movie-dub test suite.

Adds the ``scripts/`` directory to ``sys.path`` so test modules can import
the pipeline modules directly (they are not an installed package).
"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
