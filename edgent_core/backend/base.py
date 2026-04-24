"""Backend protocols and deterministic stubs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence

from ..types import ImageInput


@dataclass(frozen=True)
class ClassifierOutput:
    """Closed-set classification result."""

    choice: str
    confidence: float


class VLMBackend(Protocol):
    """Vision-language backend interface."""

    name: str

    def generate(self, image: ImageInput, query: str, **kwargs: object) -> str:
        """Produce a text response for the supplied image and query."""


class LLMBackend(Protocol):
    """Language backend interface."""

    name: str

    def classify(
        self,
        instruction: str,
        choices: list[str],
        **kwargs: object,
    ) -> ClassifierOutput:
        """Pick one choice from a closed set."""

    def generate(self, instruction: str, **kwargs: object) -> str:
        """Produce a free-form text response."""


class StubLLMBackend:
    """Deterministic classifier and generator for local tests."""

    name = "stub-llm"

    def classify(
        self,
        instruction: str,
        choices: list[str],
        **kwargs: object,
    ) -> ClassifierOutput:
        """Choose the first matching choice or hint from the instruction."""

        normalized = instruction.lower()
        choice_hints = kwargs.get("choice_hints") or {}
        if not isinstance(choice_hints, Mapping):
            choice_hints = {}

        for choice in choices:
            if choice.lower() in normalized:
                return ClassifierOutput(choice=choice, confidence=0.95)

            raw_hints = choice_hints.get(choice, [])
            hints = raw_hints if isinstance(raw_hints, Sequence) else []
            for hint in hints:
                if isinstance(hint, str) and hint and hint.lower() in normalized:
                    return ClassifierOutput(choice=choice, confidence=0.9)

        if not choices:
            return ClassifierOutput(choice="", confidence=0.0)
        return ClassifierOutput(choice=choices[0], confidence=0.5)

    def generate(self, instruction: str, **kwargs: object) -> str:
        """Return the input instruction unchanged."""

        return instruction


class StubVLMBackend:
    """Deterministic image-to-text backend for local tests."""

    name = "stub-vlm"

    def generate(self, image: ImageInput, query: str, **kwargs: object) -> str:
        """Return a predictable text response without external calls."""

        del image, kwargs
        cleaned = query.strip() if query else "generated text"
        return f"{self.name}:{cleaned}"
