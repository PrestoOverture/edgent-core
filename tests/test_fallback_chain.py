from __future__ import annotations

import time

from edgent_core import FallbackChain, FallbackTrigger, RouteDecision, RouteError


class StaticRouter:
    def __init__(self, decision: RouteDecision) -> None:
        self.decision = decision

    def route(self, input) -> RouteDecision:  # noqa: ANN001
        del input
        return self.decision


class ErrorRouter:
    def route(self, input) -> RouteDecision:  # noqa: ANN001
        del input
        raise RouteError("boom")


class SlowRouter:
    def __init__(self, decision: RouteDecision, delay: float) -> None:
        self.decision = decision
        self.delay = delay

    def route(self, input) -> RouteDecision:  # noqa: ANN001
        del input
        time.sleep(self.delay)
        return self.decision


def test_fallback_chain_on_exception() -> None:
    fallback = StaticRouter(
        RouteDecision(
            intent="fallback",
            confidence=0.9,
            secondary_intent=None,
            rationale="fallback",
            routed_tool=None,
        )
    )
    chain = FallbackChain(
        primary=ErrorRouter(),
        fallback=fallback,
        trigger=FallbackTrigger(on_timeout_ms=None, on_low_confidence=None, on_exception=True),
    )

    decision = chain.route(input=None)

    assert decision.intent == "fallback"


def test_fallback_chain_on_low_confidence() -> None:
    primary = StaticRouter(
        RouteDecision(
            intent="primary",
            confidence=0.2,
            secondary_intent=None,
            rationale="primary",
            routed_tool=None,
        )
    )
    fallback = StaticRouter(
        RouteDecision(
            intent="fallback",
            confidence=0.8,
            secondary_intent=None,
            rationale="fallback",
            routed_tool=None,
        )
    )
    chain = FallbackChain(
        primary=primary,
        fallback=fallback,
        trigger=FallbackTrigger(on_timeout_ms=None, on_low_confidence=0.5, on_exception=False),
    )

    decision = chain.route(input=None)

    assert decision.intent == "fallback"


def test_fallback_chain_on_timeout() -> None:
    primary = SlowRouter(
        RouteDecision(
            intent="primary",
            confidence=0.9,
            secondary_intent=None,
            rationale="primary",
            routed_tool=None,
        ),
        delay=0.02,
    )
    fallback = StaticRouter(
        RouteDecision(
            intent="fallback",
            confidence=0.9,
            secondary_intent=None,
            rationale="fallback",
            routed_tool=None,
        )
    )
    chain = FallbackChain(
        primary=primary,
        fallback=fallback,
        trigger=FallbackTrigger(on_timeout_ms=1, on_low_confidence=None, on_exception=False),
    )

    decision = chain.route(input=None)

    assert decision.intent == "fallback"
