from __future__ import annotations

import httpx
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


def test_openai_llm_backend_parses_secondary_intent() -> None:
    client = _Client(['{"intent":"read_text","confidence":0.9,"secondary_intent":"menu","rationale":"menu reading"}'])
    backend = OpenAILLMBackend(api_key="key", base_url="https://example.test/v1", model="model", client=client)

    output = backend.classify(instruction="read the menu", choices=["read_text", "ask_scene"])

    assert output.choice == "read_text"
    assert output.secondary_intent == "menu"
    assert output.confidence == 0.9


def test_openai_llm_backend_handles_null_secondary_intent() -> None:
    client = _Client(['{"intent":"ask_scene","confidence":0.85,"secondary_intent":null,"rationale":"scene query"}'])
    backend = OpenAILLMBackend(api_key="key", base_url="https://example.test/v1", model="model", client=client)

    output = backend.classify(instruction="what is this", choices=["read_text", "ask_scene"])

    assert output.choice == "ask_scene"
    assert output.secondary_intent is None


def test_openai_llm_backend_passes_secondary_choices_to_prompt() -> None:
    client = _Client(['{"intent":"read_text","confidence":0.9,"secondary_intent":"menu","rationale":""}'])
    backend = OpenAILLMBackend(api_key="key", base_url="https://example.test/v1", model="model", client=client)

    backend.classify(
        instruction="read the menu",
        choices=["read_text", "ask_scene"],
        secondary_choices={"read_text": ["menu", "signboard"]},
    )

    # Verify secondary_choices was included in the prompt
    call_kwargs = client.chat.completions.calls[0]
    user_content = call_kwargs["messages"][1]["content"]
    import json
    payload = json.loads(user_content)
    assert "secondary_choices" in payload
    assert payload["secondary_choices"]["read_text"] == ["menu", "signboard"]


def test_openai_llm_backend_logprobs_disabled_by_default() -> None:
    client = _Client(['{"intent":"alpha","confidence":0.9,"rationale":""}'])
    backend = OpenAILLMBackend(api_key="key", base_url="https://example.test/v1", model="model", client=client)

    backend.classify(instruction="test", choices=["alpha", "beta"])

    assert backend.last_calibrated_confidence is None
    # Should not have passed logprobs=True
    assert "logprobs" not in client.chat.completions.calls[0]


def test_openai_llm_backend_calibrated_confidence_computation() -> None:
    """Test _compute_calibrated_confidence with synthetic logprobs."""

    class _FakeTokenLogprob:
        def __init__(self, token: str, logprob: float) -> None:
            self.token = token
            self.logprob = logprob
            self.top_logprobs = []

    client = _Client(['{"intent":"alpha","confidence":0.9,"rationale":""}'])
    backend = OpenAILLMBackend(api_key="key", base_url="https://example.test/v1", model="model", client=client)

    # Simulate logprobs where "alpha" tokens have moderate confidence
    # Tokens: ['{"', 'intent', '":"', 'alpha', '",', ...]
    logprobs = [
        _FakeTokenLogprob('{"', 0.0),
        _FakeTokenLogprob('intent', 0.0),
        _FakeTokenLogprob('":"', 0.0),
        _FakeTokenLogprob('alpha', -0.5),  # log(-0.5) → exp(-0.5) ≈ 0.607
        _FakeTokenLogprob('",', 0.0),
    ]
    backend._compute_calibrated_confidence("alpha", logprobs)
    assert backend.last_calibrated_confidence is not None
    assert 0.5 < backend.last_calibrated_confidence < 0.7  # ~0.607


def test_openai_llm_backend_calibrated_confidence_exact_match() -> None:
    """Test calibrated confidence when intent tokens are perfectly confident."""

    class _FakeTokenLogprob:
        def __init__(self, token: str, logprob: float) -> None:
            self.token = token
            self.logprob = logprob
            self.top_logprobs = []

    client = _Client(['{"intent":"beta","confidence":0.9,"rationale":""}'])
    backend = OpenAILLMBackend(api_key="key", base_url="https://example.test/v1", model="model", client=client)

    logprobs = [
        _FakeTokenLogprob('{"', 0.0),
        _FakeTokenLogprob('intent', 0.0),
        _FakeTokenLogprob('":"', 0.0),
        _FakeTokenLogprob('beta', 0.0),  # logprob 0.0 = probability 1.0
        _FakeTokenLogprob('",', 0.0),
    ]
    backend._compute_calibrated_confidence("beta", logprobs)
    assert backend.last_calibrated_confidence == 1.0


def test_openai_llm_backend_calibrated_confidence_choice_not_found() -> None:
    """Test that calibrated confidence is None when choice not found in tokens."""

    class _FakeTokenLogprob:
        def __init__(self, token: str, logprob: float) -> None:
            self.token = token
            self.logprob = logprob
            self.top_logprobs = []

    client = _Client(['{"intent":"alpha","confidence":0.9,"rationale":""}'])
    backend = OpenAILLMBackend(api_key="key", base_url="https://example.test/v1", model="model", client=client)

    # Choice "gamma" doesn't appear in tokens
    logprobs = [
        _FakeTokenLogprob('alpha', 0.0),
    ]
    backend._compute_calibrated_confidence("gamma", logprobs)
    assert backend.last_calibrated_confidence is None


def test_openai_llm_backend_normalizes_choice_whitespace_and_case() -> None:
    """Choice is strip().lower()-normalized before matching (C6 micro-optimization)."""
    client = _Client(['{"intent":"  Read_Text ","confidence":0.9,"rationale":"ok"}'])
    backend = OpenAILLMBackend(api_key="key", base_url="https://example.test/v1", model="model", client=client)

    output = backend.classify(instruction="read menu", choices=["read_text", "ask_scene"])

    assert output.choice == "read_text"
    assert output.confidence == 0.9


def test_openai_llm_backend_wraps_timeout_as_backend_error() -> None:
    """API timeout is wrapped in BackendError after retries exhausted."""
    from openai import APITimeoutError

    class _TimeoutCompletions:
        def create(self, **kwargs):  # noqa: ANN003
            raise APITimeoutError(request=httpx.Request("POST", "https://example.test"))

    class _TimeoutChat:
        def __init__(self) -> None:
            self.completions = _TimeoutCompletions()

    class _TimeoutClient:
        def __init__(self) -> None:
            self.chat = _TimeoutChat()

    backend = OpenAILLMBackend(api_key="key", base_url="https://example.test/v1", model="model", client=_TimeoutClient())

    with pytest.raises(BackendError, match="timed out"):
        backend.classify(instruction="test", choices=["a"])


def test_openai_llm_backend_wraps_connection_error_as_backend_error() -> None:
    """API connection error is wrapped in BackendError after retries exhausted."""
    from openai import APIConnectionError

    class _ConnCompletions:
        def create(self, **kwargs):  # noqa: ANN003
            raise APIConnectionError(request=httpx.Request("POST", "https://example.test"))

    class _ConnChat:
        def __init__(self) -> None:
            self.completions = _ConnCompletions()

    class _ConnClient:
        def __init__(self) -> None:
            self.chat = _ConnChat()

    backend = OpenAILLMBackend(api_key="key", base_url="https://example.test/v1", model="model", client=_ConnClient())

    with pytest.raises(BackendError, match="connection failed"):
        backend.classify(instruction="test", choices=["a"])
