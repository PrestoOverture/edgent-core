from __future__ import annotations

import pytest

from edgent_core import BackendError, OpenAILLMBackend


class _Message:
    def __init__(self, content: str) -> None:
        self.content = content


class _Choice:
    def __init__(self, content: str) -> None:
        self.message = _Message(content)


class _Response:
    def __init__(self, content: str) -> None:
        self.choices = [_Choice(content)]


class _Completions:
    def __init__(self, responses: list[object]) -> None:
        self.responses = responses
        self.calls: list[dict] = []

    def create(self, **kwargs) -> _Response:  # noqa: ANN003
        self.calls.append(kwargs)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return _Response(str(response))


class _Chat:
    def __init__(self, responses: list[object]) -> None:
        self.completions = _Completions(responses)


class _Client:
    def __init__(self, responses: list[object]) -> None:
        self.chat = _Chat(responses)


def test_openai_llm_backend_parses_json_classifier_response() -> None:
    client = _Client(['{"intent":"beta","confidence":0.82,"rationale":"matched beta"}'])
    backend = OpenAILLMBackend(api_key="key", base_url="https://example.test/v1", model="model", client=client)

    output = backend.classify(
        instruction="route this",
        choices=["alpha", "beta"],
        choice_hints={"alpha": ["first"], "beta": ["second"]},
    )

    assert output.choice == "beta"
    assert output.confidence == 0.82
    assert backend.last_rationale == "matched beta"
    assert client.chat.completions.calls[0]["response_format"] == {"type": "json_object"}


def test_openai_llm_backend_retries_without_json_mode_for_non_json_response() -> None:
    client = _Client(["not json", '{"intent":"alpha","confidence":0.7,"rationale":"fallback"}'])
    backend = OpenAILLMBackend(api_key="key", base_url="https://example.test/v1", model="model", client=client)

    output = backend.classify(instruction="route this", choices=["alpha", "beta"])

    assert output.choice == "alpha"
    assert output.confidence == 0.7
    assert len(client.chat.completions.calls) == 2
    assert "response_format" in client.chat.completions.calls[0]
    assert "response_format" not in client.chat.completions.calls[1]


def test_openai_llm_backend_raises_when_retry_still_fails() -> None:
    client = _Client(["not json", "still not json"])
    backend = OpenAILLMBackend(api_key="key", base_url="https://example.test/v1", model="model", client=client)

    with pytest.raises(BackendError):
        backend.classify(instruction="route this", choices=["alpha", "beta"])
