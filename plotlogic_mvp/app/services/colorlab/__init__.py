"""colorlab - glyph-first curve extraction, run side by side with the legacy pipeline.

Reads an existing plotlogic run directory; writes only into ``<run_dir>/colorlab/``.
"""

from .config import ColorLabParams
from .runner import run_colorlab

__all__ = ["ColorLabParams", "run_colorlab"]
