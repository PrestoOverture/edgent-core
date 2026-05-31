"""OpenAI-compatible text backend for closed-set classification."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

from openai import APIConnectionError, APITimeoutError, OpenAI

from ..errors import BackendError
from ..types import Usage
from .base import ClassifierOutput

_CLASSIFIER_SYSTEM_PROMPT = """You are a closed-set routing classifier.
Pick exactly one id from the supplied choices. Do not invent ids.
If secondary_choices are provided and the selected intent has secondary options, pick the best matching secondary_choice id. Otherwise set secondary_intent to null.
Return only JSON with this shape:
{"intent":"<one choice id>","confidence":0.0,"secondary_intent":null or "<secondary id>","rationale":"short reason"}
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
        max_tokens: int = 128,
        client: object | None = None,
        enable_logprobs: bool = False,
        request_timeout: float = 30.0,
        max_retries: int = 2,
    ) -> None:
        self.name = name or model
        self.model = model
        self.max_tokens = max_tokens
        self.last_rationale: str | None = None
        self.last_usage: Usage = Usage()
        self.last_calibrated_confidence: float | None = None
        self._enable_logprobs = enable_logprobs
        self._client = client or OpenAI(
            api_key=api_key, base_url=base_url, timeout=request_timeout, max_retries=max_retries,
        )

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
            secondary_choices=kwargs.get("secondary_choices"),
        )
        messages: list[dict[str, object]] = [
            {"role": "system", "content": _CLASSIFIER_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        # Try with logprobs first for calibrated confidence (if enabled)
        self.last_calibrated_confidence = None
        if self._enable_logprobs:
            try:
                content, logprobs = self._chat_with_logprobs(messages, response_format={"type": "json_object"})
                result = self._parse_classifier_output(content, choices)
                self._compute_calibrated_confidence(result.choice, logprobs)
                return result
            except Exception:
                pass

            try:
                content, logprobs = self._chat_with_logprobs(messages)
                result = self._parse_classifier_output(content, choices)
                self._compute_calibrated_confidence(result.choice, logprobs)
                return result
            except Exception:
                pass

        # Standard path (no logprobs)
        # Attempt 1: with JSON response_format
        result_content: str | None = None
        try:
            result_content = self._chat(messages, response_format={"type": "json_object"})
        except BackendError:
            raise  # timeout / connection — don't swallow
        except Exception:
            result_content = None

        if result_content is not None:
            try:
                return self._parse_classifier_output(result_content, choices)
            except Exception:
                pass  # parse failure — retry without json mode

        # Attempt 2: without JSON response_format
        try:
            result_content = self._chat(messages)
        except BackendError:
            raise  # timeout / connection — don't swallow
        except Exception as exc:
            raise BackendError("LLM classification failed.") from exc
        return self._parse_classifier_output(result_content, choices)

    def generate(self, instruction: str, **kwargs: object) -> str:
        """Generate free-form text from an instruction."""

        del kwargs
        response = self._chat([{"role": "user", "content": instruction}])
        return response

    def _call_api(self, **kwargs: object) -> Any:
        """Call the chat completions API with timeout/retry error wrapping."""
        try:
            return self._client.chat.completions.create(**kwargs)  # type: ignore[attr-defined]  # duck-typed client for test injection
        except APITimeoutError as exc:
            raise BackendError(f"LLM request timed out: {exc}") from exc
        except APIConnectionError as exc:
            raise BackendError(f"LLM connection failed: {exc}") from exc

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
        response = self._call_api(**kwargs)
        usage = getattr(response, "usage", None)
        if usage is not None:
            self.last_usage = Usage(
                prompt_tokens=getattr(usage, "prompt_tokens", 0),
                completion_tokens=getattr(usage, "completion_tokens", 0),
            )
        return response.choices[0].message.content or ""

    def _chat_with_logprobs(
        self,
        messages: list[dict[str, object]],
        response_format: dict[str, str] | None = None,
    ) -> tuple[str, list[object]]:
        """Like _chat but also returns token logprobs. Raises if logprobs unavailable."""

        kwargs: dict[str, object] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "logprobs": True,
            "top_logprobs": 5,
        }
        if response_format is not None:
            kwargs["response_format"] = response_format
        response = self._call_api(**kwargs)
        usage = getattr(response, "usage", None)
        if usage is not None:
            self.last_usage = Usage(
                prompt_tokens=getattr(usage, "prompt_tokens", 0),
                completion_tokens=getattr(usage, "completion_tokens", 0),
            )
        choice = response.choices[0]
        logprobs_obj = getattr(choice, "logprobs", None)
        if logprobs_obj is None:
            raise BackendError("Logprobs not available in response.")
        content = choice.message.content or ""
        token_logprobs = getattr(logprobs_obj, "content", None)
        if not token_logprobs:
            raise BackendError("Logprobs content is empty.")
        return content, token_logprobs

    def _compute_calibrated_confidence(self, choice: str, token_logprobs: list[object]) -> None:
        """Compute calibrated confidence from token logprobs for the chosen intent.

        Finds the tokens that spell the chosen intent in the output and computes
        the average probability of those tokens.
        """

        # Collect all tokens and their logprobs
        tokens: list[str] = []
        logprobs: list[float] = []
        for tl in token_logprobs:
            token = getattr(tl, "token", "")
            lp = getattr(tl, "logprob", -999.0)
            tokens.append(token)
            logprobs.append(lp)

        # Find the substring of all tokens concatenated
        full_text = "".join(tokens)

        # Find where the choice string appears in the token stream
        choice_lower = choice.lower()
        char_pos = full_text.lower().find(choice_lower)
        if char_pos < 0:
            # Choice not found in output tokens; can't calibrate
            return

        # Map character positions back to token indices
        token_start = None
        token_end = None
        cum_pos = 0
        for i, token in enumerate(tokens):
            token_end_pos = cum_pos + len(token)
            if token_start is None and char_pos < token_end_pos:
                token_start = i
            if token_start is not None and char_pos + len(choice_lower) <= token_end_pos:
                token_end = i + 1
                break
            cum_pos = token_end_pos

        if token_start is None or token_end is None:
            return

        # Average logprob of the intent tokens
        intent_logprobs = logprobs[token_start:token_end]
        if not intent_logprobs:
            return
        avg_logprob = sum(intent_logprobs) / len(intent_logprobs)
        # Clamp to avoid math domain issues
        avg_logprob = max(avg_logprob, -20.0)
        calibrated = math.exp(avg_logprob)
        self.last_calibrated_confidence = max(0.0, min(calibrated, 1.0))

    def _parse_classifier_output(self, content: str, choices: list[str]) -> ClassifierOutput:
        try:
            payload = json.loads(content)
        except json.JSONDecodeError as exc:
            raise BackendError("LLM returned non-JSON classifier output.") from exc
        if not isinstance(payload, Mapping):
            raise BackendError("LLM classifier output must be a JSON object.")

        raw_choice = payload.get("intent")
        if not isinstance(raw_choice, str):
            raise BackendError("LLM returned an unknown classifier choice.")
        choice = raw_choice.strip().lower()
        if choice not in choices:
            raise BackendError("LLM returned an unknown classifier choice.")

        raw_confidence = payload.get("confidence")
        if not isinstance(raw_confidence, (int, float)):
            raise BackendError("LLM classifier confidence must be numeric.")
        confidence = max(0.0, min(float(raw_confidence), 1.0))

        rationale = payload.get("rationale")
        self.last_rationale = rationale if isinstance(rationale, str) else None

        secondary_intent_raw = payload.get("secondary_intent")
        if isinstance(secondary_intent_raw, str):
            secondary_intent = secondary_intent_raw.strip().lower() or None
        else:
            secondary_intent = None

        return ClassifierOutput(choice=choice, confidence=confidence, secondary_intent=secondary_intent)


def _build_classifier_user_prompt(
    *,
    instruction: str,
    choices: list[str],
    choice_hints: object,
    secondary_choices: object = None,
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

    payload: dict[str, object] = {
        "instruction": instruction,
        "choices": choices,
        "choice_hints": hints,
    }
    if isinstance(secondary_choices, Mapping) and secondary_choices:
        payload["secondary_choices"] = secondary_choices
    return json.dumps(payload, ensure_ascii=False)
