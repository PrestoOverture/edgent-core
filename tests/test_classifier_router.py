from __future__ import annotations

from edgent_core import (
    ClassifierOutput,
    ClassifierRouter,
    IntentNode,
    RuntimeInput,
    StubLLMBackend,
    TaxonomySchema,
)


def test_classifier_router_returns_deterministic_primary_intent() -> None:
    taxonomy = TaxonomySchema(
        primary_nodes=[
            IntentNode(id="alpha", label="Alpha", parent_id=None, routed_tool="tool_alpha"),
            IntentNode(id="beta", label="Beta", parent_id=None, routed_tool="tool_beta"),
        ],
        secondary_nodes_by_parent={},
        tool_suggestions_by_intent={"alpha": "tool_alpha", "beta": "tool_beta"},
        clarify_threshold=0.75,
        fallback_threshold=0.55,
    )
    router = ClassifierRouter(taxonomy=taxonomy, llm_backend=StubLLMBackend())

    decision = router.route(
        RuntimeInput(text="please route this to beta", image=None, audio=None, session_id="s1")
    )

    assert decision.intent == "beta"
    assert decision.routed_tool == "tool_beta"
    assert decision.confidence >= 0.9


class _LLMWithSecondary:
    """Fake LLM that returns secondary_intent."""
    name = "fake-llm-secondary"
    last_usage = None

    def __init__(self, secondary: str | None = None) -> None:
        self._secondary = secondary

    def classify(self, instruction: str, choices: list[str], **kwargs: object) -> ClassifierOutput:
        return ClassifierOutput(choice=choices[0], confidence=0.95, secondary_intent=self._secondary)

    def generate(self, instruction: str, **kwargs: object) -> str:
        return ""


def test_classifier_router_uses_llm_secondary_intent():
    """When LLM returns a valid secondary_intent, router uses it."""
    taxonomy = TaxonomySchema(
        primary_nodes=[
            IntentNode(id="read_text", label="Read Text", parent_id=None, routed_tool="read_text"),
            IntentNode(id="ask_scene", label="Ask Scene", parent_id=None, routed_tool="describe_scene"),
        ],
        secondary_nodes_by_parent={
            "read_text": [
                IntentNode(id="menu", label="Menu", parent_id="read_text", routed_tool="read_text"),
                IntentNode(id="signboard", label="Signboard", parent_id="read_text", routed_tool="read_text"),
            ],
        },
        tool_suggestions_by_intent={
            "read_text": "read_text",
            "ask_scene": "describe_scene",
            "menu": "read_text",
            "signboard": "read_text",
        },
        clarify_threshold=0.75,
        fallback_threshold=0.55,
    )
    llm = _LLMWithSecondary(secondary="menu")
    router = ClassifierRouter(taxonomy=taxonomy, llm_backend=llm)

    decision = router.route(
        RuntimeInput(text="read the menu", image=None, audio=None, session_id="s1")
    )

    assert decision.intent == "read_text"
    assert decision.secondary_intent == "menu"


def test_classifier_router_rejects_invalid_llm_secondary():
    """When LLM returns an invalid secondary_intent, router falls back to substring match."""
    taxonomy = TaxonomySchema(
        primary_nodes=[
            IntentNode(id="read_text", label="Read Text", parent_id=None, routed_tool="read_text"),
        ],
        secondary_nodes_by_parent={
            "read_text": [
                IntentNode(id="menu", label="Menu", parent_id="read_text", routed_tool="read_text"),
            ],
        },
        tool_suggestions_by_intent={"read_text": "read_text", "menu": "read_text"},
        clarify_threshold=0.75,
        fallback_threshold=0.55,
    )
    # LLM returns "invalid_secondary" which is not a valid secondary node
    llm = _LLMWithSecondary(secondary="invalid_secondary")
    router = ClassifierRouter(taxonomy=taxonomy, llm_backend=llm)

    decision = router.route(
        RuntimeInput(text="read the menu", image=None, audio=None, session_id="s1")
    )

    assert decision.intent == "read_text"
    # Should fall back to substring match which finds "menu" in "read the menu"
    assert decision.secondary_intent == "menu"


def test_classifier_router_no_secondary_when_none_from_llm():
    """When LLM returns None secondary_intent and no substring match, result is None."""
    taxonomy = TaxonomySchema(
        primary_nodes=[
            IntentNode(id="read_text", label="Read Text", parent_id=None, routed_tool="read_text"),
        ],
        secondary_nodes_by_parent={
            "read_text": [
                IntentNode(id="menu", label="Menu", parent_id="read_text", routed_tool="read_text"),
            ],
        },
        tool_suggestions_by_intent={"read_text": "read_text", "menu": "read_text"},
        clarify_threshold=0.75,
        fallback_threshold=0.55,
    )
    llm = _LLMWithSecondary(secondary=None)
    router = ClassifierRouter(taxonomy=taxonomy, llm_backend=llm)

    decision = router.route(
        RuntimeInput(text="something unrelated", image=None, audio=None, session_id="s1")
    )

    assert decision.intent == "read_text"
    assert decision.secondary_intent is None


def test_classifier_output_has_secondary_intent():
    """ClassifierOutput now includes secondary_intent field."""
    output = ClassifierOutput(choice="read_text", confidence=0.95, secondary_intent="menu")
    assert output.secondary_intent == "menu"

    output_none = ClassifierOutput(choice="read_text", confidence=0.95)
    assert output_none.secondary_intent is None
