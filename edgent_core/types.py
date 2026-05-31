"""Transport-agnostic media input containers and cost primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .errors import ValidationError


@dataclass(frozen=True)
class Usage:
    """Token usage from one backend call."""

    prompt_tokens: int = 0
    completion_tokens: int = 0


@runtime_checkable
class CostCalculator(Protocol):
    """Compute USD cost from token usage. Adapter-supplied, not core-internal."""

    def calculate(self, usage: Usage) -> float:
        """Return cost in USD for the given token counts."""


@dataclass(frozen=True)
class ImageInput:
    """Image container for runtime calls.

    Phase 1 supports in-memory bytes and optional filesystem paths.
    """

    data: bytes | None = None
    path: str | None = None

    def __post_init__(self) -> None:
        if self.data is None and self.path is None:
            raise ValidationError("ImageInput requires bytes or a path.")
        if self.data is not None and not isinstance(self.data, (bytes, bytearray)):
            raise ValidationError("ImageInput.data must be bytes-like.")

    def persistent_ref(self) -> str | None:
        """Return the stable reference used by trace logging."""

        return self.path


@dataclass(frozen=True)
class VisualCue:
    """Lightweight visual hint injected into routing decisions."""

    has_readable_text: bool
    text_surface_type: str | None = None
    visual_confidence: float = 1.0


@dataclass(frozen=True)
class AudioInput:
    """Audio container for runtime calls.

    Phase 1 supports in-memory bytes and optional filesystem paths.
    """

    data: bytes | None = None
    path: str | None = None

    def __post_init__(self) -> None:
        if self.data is None and self.path is None:
            raise ValidationError("AudioInput requires bytes or a path.")
        if self.data is not None and not isinstance(self.data, (bytes, bytearray)):
            raise ValidationError("AudioInput.data must be bytes-like.")

    def persistent_ref(self) -> str | None:
        """Return the stable reference used by trace logging."""

        return self.path
