"""OpenAI-compatible text backend for closed-set classification."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

from openai import OpenAI

from ..errors import BackendError
from .base import ClassifierOutput


_CLASSIFIER_SYSTEM_PROMPT = """You are a closed-set routing classifier.
Pick exactly one id from the supplied choices. Do not invent ids.
Return only JSON with this shape:
{"intent":"<one choice id>","confidence":0.0,"secondary_intent":null,"rationale":"short reason"}
The confidence must be a number between 0.0 and 1.0."""


class OpenAILLMBackend:
    """LLM backend that calls any OpenAI-compatible chat-completion API."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        name: str | None = None,
        max_tokens: int = 256,
        client: object | None = None,
    ) -> None:
        self.name = name or model
        self.model = model
        self.max_tokens = max_tokens
        self.last_rationale: str | None = None
        self._client = client or OpenAI(api_key=api_key, base_url=base_url)

    def classify(
        self,
        instruction: str,
        choices: list[str],
        **kwargs: object,
    ) -> ClassifierOutput:
        """Classify text into one of the supplied choices."""

        if not choices:
            return ClassifierOutput(choice="", confidence=0.0)

        user_prompt = _build_classifier_user_prompt(
            instruction=instruction,
            choices=choices,
            choice_hints=kwargs.get("choice_hints"),
        )
        messages = [
            {"role": "system", "content": _CLASSIFIER_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        try:
            content = self._chat(messages, response_format={"type": "json_object"})
            return self._parse_classifier_output(content, choices)
        except Exception as first_error:
            try:
                content = self._chat(messages)
                return self._parse_classifier_output(content, choices)
            except Exception as second_error:
                raise BackendError("LLM classification failed.") from second_error or first_error

    def generate(self, instruction: str, **kwargs: object) -> str:
        """Generate free-form text from an instruction."""

        del kwargs
        response = self._chat([{"role": "user", "content": instruction}])
        return response

    def _chat(
        self,
        messages: list[dict[str, object]],
        response_format: dict[str, str] | None = None,
    ) -> str:
        kwargs: dict[str, object] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
        }
        if response_format is not None:
            kwargs["response_format"] = response_format
        response = self._client.chat.completions.create(**kwargs)  # type: ignore[attr-defined]
        return response.choices[0].message.content or ""

    def _parse_classifier_output(self, content: str, choices: list[str]) -> ClassifierOutput:
        try:
            payload = json.loads(content)
        except json.JSONDecodeError as exc:
            raise BackendError("LLM returned non-JSON classifier output.") from exc
        if not isinstance(payload, Mapping):
            raise BackendError("LLM classifier output must be a JSON object.")

        choice = payload.get("intent")
        if not isinstance(choice, str) or choice not in choices:
            raise BackendError("LLM returned an unknown classifier choice.")

        raw_confidence = payload.get("confidence")
        if not isinstance(raw_confidence, (int, float)):
            raise BackendError("LLM classifier confidence must be numeric.")
        confidence = max(0.0, min(float(raw_confidence), 1.0))

        rationale = payload.get("rationale")
        self.last_rationale = rationale if isinstance(rationale, str) else None
        return ClassifierOutput(choice=choice, confidence=confidence)


def _build_classifier_user_prompt(
    *,
    instruction: str,
    choices: list[str],
    choice_hints: object,
) -> str:
    hints: dict[str, list[str]] = {}
    if isinstance(choice_hints, Mapping):
        for choice in choices:
            raw_hint = choice_hints.get(choice)
            if isinstance(raw_hint, Sequence) and not isinstance(raw_hint, str):
                hints[choice] = [item for item in raw_hint if isinstance(item, str)]
            elif isinstance(raw_hint, str):
                hints[choice] = [raw_hint]
            else:
                hints[choice] = []
    else:
        hints = {choice: [] for choice in choices}

    return json.dumps(
        {
            "instruction": instruction,
            "choices": choices,
            "choice_hints": hints,
        },
        ensure_ascii=False,
    )
