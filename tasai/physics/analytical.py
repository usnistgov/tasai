"""Compatibility module exposing analytical models without Sunny import hassles."""
from __future__ import annotations

from tasai.sunny import SquareLatticeFM, NNOnlyModel  # type: ignore

__all__ = ["SquareLatticeFM", "NNOnlyModel"]
