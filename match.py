"""Headless match API.

The implementation lives alongside the viewer for now so it can reuse the
existing, tested rule methods without changing their behavior.  Importing
``Match`` from this small module keeps training code independent of the UI
module's class layout.
"""

from arena_viewer import Match

__all__ = ["Match"]
