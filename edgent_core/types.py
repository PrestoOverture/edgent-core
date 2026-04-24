"""Transport-agnostic media input containers."""

from __future__ import annotations

from dataclasses import dataclass

from .errors import ValidationError


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
