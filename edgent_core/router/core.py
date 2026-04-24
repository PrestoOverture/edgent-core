"""Routing primitives and reference implementations."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Protocol

from ..backend import LLMBackend
from ..errors import RouteError, ValidationError

if TYPE_CHECKING:
    from ..orchestrator import RuntimeInput


_SNAKE_CASE = re.compile(r"^[a-z][a-z0-9_]*$")


@dataclass(frozen=True)
class IntentNode:
    """Intent node consumed by closed-set routing."""

    id: str
    label: str
    parent_id: str | None
    routed_tool: str | None
    description: str | None = None

    def __post_init__(self) -> None:
        if not _SNAKE_CASE.match(self.id):
            raise ValidationError("IntentNode.id must be snake_case.")


@dataclass(frozen=True)
class TaxonomySchema:
    """Stable taxonomy contract injected by an adapter."""

    primary_nodes: list[IntentNode]
    secondary_nodes_by_parent: dict[str, list[IntentNode]] = field(default_factory=dict)
    tool_suggestions_by_intent: dict[str, str | None] = field(default_factory=dict)
    clarify_threshold: float = 0.75
    fallback_threshold: float = 0.55

    def __post_init__(self) -> None:
        if not 0.0 <= self.clarify_threshold <= 1.0:
            raise ValidationError("clarify_threshold must be within [0.0, 1.0].")
        if not 0.0 <= self.fallback_threshold <= 1.0:
            raise ValidationError("fallback_threshold must be within [0.0, 1.0].")
        if self.clarify_threshold < self.fallback_threshold:
            raise ValidationError("clarify_threshold must be >= fallback_threshold.")

        seen_ids: set[str] = set()
        primary_ids = set()
        for node in self.primary_nodes:
            if node.parent_id is not None:
                raise ValidationError("Primary nodes must have parent_id=None.")
            if node.id in seen_ids:
                raise ValidationError("Intent ids must be globally unique.")
            seen_ids.add(node.id)
            primary_ids.add(node.id)

        for parent_id, nodes in self.secondary_nodes_by_parent.items():
            if parent_id not in primary_ids:
                raise ValidationError("Secondary nodes must attach to an existing primary node.")
            for node in nodes:
                if node.parent_id != parent_id:
                    raise ValidationError("Secondary node parent_id must match its map key.")
                if node.id in seen_ids:
                    raise ValidationError("Intent ids must be globally unique.")
                seen_ids.add(node.id)

        if not set(self.tool_suggestions_by_intent).issubset(seen_ids):
            raise ValidationError("tool_suggestions_by_intent keys must reference declared ids.")

        for node in self.primary_nodes + self.secondary_nodes():
            if node.id in self.tool_suggestions_by_intent:
                expected = self.tool_suggestions_by_intent[node.id]
                if node.routed_tool != expected:
                    raise ValidationError("IntentNode.routed_tool must match tool_suggestions_by_intent.")

    def secondary_nodes(self) -> list[IntentNode]:
        """Return every declared secondary node."""

        return [node for nodes in self.secondary_nodes_by_parent.values() for node in nodes]

    def primary_by_id(self, intent_id: str) -> IntentNode:
        """Return a primary node by id."""

        for node in self.primary_nodes:
            if node.id == intent_id:
                return node
        raise RouteError(f"Unknown primary intent: {intent_id}")


@dataclass(frozen=True)
class RouteDecision:
    """Closed-set routing output."""

    intent: str
    confidence: float
    secondary_intent: str | None
    rationale: str | None
    routed_tool: str | None


class Router(Protocol):
    """Router interface."""

    def route(self, input: RuntimeInput) -> RouteDecision:
        """Route one normalized runtime input."""


@dataclass(frozen=True)
class HookOutcome:
    """Result of evaluating a confidence hook."""

    action: Literal["pass", "clarify", "fallback"]
    clarify_prompt: str | None
    fallback_tool: str | None


class ConfidenceHook(Protocol):
    """Hook invoked after routing and before tool execution."""

    def on_route(self, decision: RouteDecision, input: RuntimeInput) -> HookOutcome:
        """Return the next action for the current route decision."""


class ClassifierRouter:
    """Reference router built on one closed-set classification call."""

    def __init__(
        self,
        taxonomy: TaxonomySchema,
        llm_backend: LLMBackend,
        hook: ConfidenceHook | None = None,
    ) -> None:
        self.taxonomy = taxonomy
        self.llm_backend = llm_backend
        self.hook = hook

    def route(self, input: RuntimeInput) -> RouteDecision:
        """Route one input into the configured taxonomy."""

        primary_nodes = self.taxonomy.primary_nodes
        if not primary_nodes:
            raise RouteError("ClassifierRouter requires at least one primary node.")

        choices = [node.id for node in primary_nodes]
        choice_hints = {
            node.id: [part for part in (node.label, node.description, node.id) if part]
            for node in primary_nodes
        }
        instruction = input.text or ""
        output = self.llm_backend.classify(
            instruction=instruction,
            choices=choices,
            choice_hints=choice_hints,
        )
        if output.choice not in choices:
            raise RouteError(f"Backend returned unknown choice: {output.choice}")

        primary_node = self.taxonomy.primary_by_id(output.choice)
        return RouteDecision(
            intent=primary_node.id,
            confidence=output.confidence,
            secondary_intent=self._match_secondary(primary_node.id, instruction),
            rationale=f"classified via {self.llm_backend.name}",
            routed_tool=primary_node.routed_tool,
        )

    def _match_secondary(self, primary_id: str, instruction: str) -> str | None:
        """Best-effort secondary match from the same input text."""

        normalized = instruction.lower()
        for node in self.taxonomy.secondary_nodes_by_parent.get(primary_id, []):
            candidates = [node.id, node.label, node.description or ""]
            if any(candidate and candidate.lower() in normalized for candidate in candidates):
                return node.id
        return None


@dataclass(frozen=True)
class FallbackTrigger:
    """Fallback conditions for chaining routers."""

    on_timeout_ms: int | None
    on_low_confidence: float | None
    on_exception: bool


class FallbackChain:
    """Router composition with deterministic downgrade rules."""

    def __init__(
        self,
        primary: Router,
        fallback: Router,
        trigger: FallbackTrigger,
    ) -> None:
        self.primary = primary
        self.fallback = fallback
        self.trigger = trigger

    def route(self, input: RuntimeInput) -> RouteDecision:
        """Route with the primary router and fall back when configured."""

        started = time.monotonic()
        try:
            primary_decision = self.primary.route(input)
        except RouteError:
            if self.trigger.on_exception:
                return self.fallback.route(input)
            raise

        elapsed_ms = (time.monotonic() - started) * 1000.0
        if self.trigger.on_timeout_ms is not None and elapsed_ms > self.trigger.on_timeout_ms:
            return self.fallback.route(input)

        if (
            self.trigger.on_low_confidence is not None
            and primary_decision.confidence < self.trigger.on_low_confidence
        ):
            return self.fallback.route(input)

        return primary_decision


class SupervisorRouter:
    """Experimental router reserved for future orchestration patterns."""

    def route(self, input: RuntimeInput) -> RouteDecision:
        """Raise until the experimental router is wired in a future phase."""

        del input
        raise NotImplementedError("SupervisorRouter is not wired in Phase 1.")
