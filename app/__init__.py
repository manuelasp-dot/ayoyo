"""Ayoyo — free, open-source facial analysis.

A Qoves-style facial analysis pipeline built entirely from permissively
licensed, free components:

  * Landmarks    -> MediaPipe Face Mesh (Apache-2.0), 478 points
  * Harmony      -> symmetry + classical facial proportions (our own math)
  * Skin         -> OpenCV color/texture heuristics (BSD), no paid API

Everything runs locally and offline. No cloud vision API required.
"""

__version__ = "0.1.0"
