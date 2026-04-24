from __future__ import annotations

from edgent_core import (
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
