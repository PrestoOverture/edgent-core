from __future__ import annotations

import pytest

from edgent_core import (
    ConfidenceHook,
    HookOutcome,
    ImageInput,
    Orchestrator,
    RouteDecision,
    RuntimeInput,
    ToolError,
    ToolRegistry,
)


class DecisionRouter:
    def __init__(self, decision: RouteDecision) -> None:
        self.decision = decision
        self.hook = None

    def route(self, input: RuntimeInput) -> RouteDecision:
        del input
        return self.decision


class ClarifyHook(ConfidenceHook):
    def on_route(self, decision: RouteDecision, input: RuntimeInput) -> HookOutcome:
        del decision, input
        return HookOutcome(action="clarify", clarify_prompt="clarify now", fallback_tool=None)


class FallbackHook(ConfidenceHook):
    def on_route(self, decision: RouteDecision, input: RuntimeInput) -> HookOutcome:
        del decision, input
        return HookOutcome(action="fallback", clarify_prompt=None, fallback_tool="tool_secondary")


class SpyTraceLogger:
    def __init__(self) -> None:
        self.events: list[str] = []
        self.hook_outcomes: list[tuple[str | None, str | None]] = []
        self.final_decisions: list[tuple[str | None, float | None]] = []

    def on_run_start(self, input: RuntimeInput) -> None:
        del input
        self.events.append("start")

    def on_route(self, decision: RouteDecision) -> None:
        del decision
        self.events.append("route")

    def on_hook_outcome(self, outcome: HookOutcome | None) -> None:
        self.events.append("hook")
        if outcome is None:
            self.hook_outcomes.append((None, None))
        else:
            self.hook_outcomes.append((outcome.action, outcome.fallback_tool))

    def on_tool_invoke(self, tool_name: str, args: dict, output: dict | None, error: Exception | None) -> None:
        del tool_name, args, output, error
        self.events.append("tool")

    def on_final_decision(self, intent: str | None, confidence: float | None) -> None:
        self.events.append("final")
        self.final_decisions.append((intent, confidence))

    def on_run_end(self, output) -> None:  # noqa: ANN001
        del output
        self.events.append("end")


class StaticTool:
    def __init__(self, name: str, text: str) -> None:
        self.name = name
        self.text = text
        self.input_schema = {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "additionalProperties": False,
        }
        self.output_schema = {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
            "additionalProperties": False,
        }

    def invoke(self, args: dict) -> dict:
        del args
        return {"text": self.text}


class FailingTool(StaticTool):
    def invoke(self, args: dict) -> dict:
        del args
        raise ToolError("tool failure")


class CapturingTool:
    def __init__(self, name: str) -> None:
        self.name = name
        self.last_args: dict | None = None
        self.input_schema = {"type": "object", "additionalProperties": True}
        self.output_schema = {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
            "additionalProperties": False,
        }

    def invoke(self, args: dict) -> dict:
        self.last_args = args
        return {"text": "captured"}


class FakeToolArgBuilder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, RuntimeInput, RouteDecision]] = []

    def build(self, tool_name: str, input: RuntimeInput, decision: RouteDecision) -> dict:
        self.calls.append((tool_name, input, decision))
        return {"text": "from builder"}


def _make_input() -> RuntimeInput:
    return RuntimeInput(text="payload", image=None, audio=None, session_id="session")


def test_orchestrator_pass_path() -> None:
    registry = ToolRegistry()
    registry.register(StaticTool("tool_primary", "pass output"))
    logger = SpyTraceLogger()
    orchestrator = Orchestrator(
        router=DecisionRouter(
            RouteDecision(
                intent="alpha",
                confidence=0.9,
                secondary_intent=None,
                rationale="pass",
                routed_tool="tool_primary",
            )
        ),
        tool_registry=registry,
        trace_logger=logger,
    )

    output = orchestrator.run(_make_input())

    assert output.text == "pass output"
    assert logger.events == ["start", "route", "hook", "tool", "final", "end"]
    assert logger.hook_outcomes == [(None, None)]
    assert logger.final_decisions == [("alpha", 0.9)]


def test_orchestrator_default_tool_arg_builder_preserves_existing_mappings() -> None:
    image = ImageInput(path="images/example.jpg")
    cases = [
        ("describe_scene", {"image": image, "query": "payload"}),
        ("read_text", {"image": image, "instruction": "payload"}),
        ("navigate_request", {"image": image, "text": "payload"}),
    ]

    for tool_name, expected_args in cases:
        registry = ToolRegistry()
        tool = CapturingTool(tool_name)
        registry.register(tool)
        orchestrator = Orchestrator(
            router=DecisionRouter(
                RouteDecision(
                    intent="alpha",
                    confidence=0.9,
                    secondary_intent=None,
                    rationale="default args",
                    routed_tool=tool_name,
                )
            ),
            tool_registry=registry,
        )

        output = orchestrator.run(
            RuntimeInput(text="payload", image=image, audio=None, session_id="session")
        )

        assert output.text == "captured"
        assert tool.last_args == expected_args


def test_orchestrator_uses_injected_tool_arg_builder() -> None:
    registry = ToolRegistry()
    registry.register(StaticTool("tool_primary", "pass output"))
    decision = RouteDecision(
        intent="alpha",
        confidence=0.9,
        secondary_intent="beta",
        rationale="custom args",
        routed_tool="tool_primary",
    )
    input = _make_input()
    builder = FakeToolArgBuilder()
    orchestrator = Orchestrator(
        router=DecisionRouter(decision),
        tool_registry=registry,
        tool_arg_builder=builder,
    )

    output = orchestrator.run(input)

    assert output.text == "pass output"
    assert builder.calls == [("tool_primary", input, decision)]


def test_orchestrator_clarify_path() -> None:
    registry = ToolRegistry()
    registry.register(StaticTool("tool_primary", "unused"))
    logger = SpyTraceLogger()
    orchestrator = Orchestrator(
        router=DecisionRouter(
            RouteDecision(
                intent="alpha",
                confidence=0.4,
                secondary_intent=None,
                rationale="clarify",
                routed_tool="tool_primary",
            )
        ),
        tool_registry=registry,
        confidence_hook=ClarifyHook(),
        trace_logger=logger,
    )

    output = orchestrator.run(_make_input())

    assert output.text == "clarify now"
    assert logger.events == ["start", "route", "hook", "final", "end"]
    assert logger.hook_outcomes == [("clarify", None)]
    assert logger.final_decisions == [(None, None)]


def test_orchestrator_fallback_path() -> None:
    registry = ToolRegistry()
    registry.register(StaticTool("tool_primary", "unused"))
    registry.register(StaticTool("tool_secondary", "fallback output"))
    logger = SpyTraceLogger()
    orchestrator = Orchestrator(
        router=DecisionRouter(
            RouteDecision(
                intent="alpha",
                confidence=0.4,
                secondary_intent=None,
                rationale="fallback",
                routed_tool="tool_primary",
            )
        ),
        tool_registry=registry,
        confidence_hook=FallbackHook(),
        trace_logger=logger,
    )

    output = orchestrator.run(_make_input())

    assert output.text == "fallback output"
    assert logger.events == ["start", "route", "hook", "tool", "final", "end"]
    assert logger.hook_outcomes == [("fallback", "tool_secondary")]
    assert logger.final_decisions == [("alpha", 0.4)]


def test_orchestrator_tool_failure_still_finishes_trace() -> None:
    registry = ToolRegistry()
    registry.register(FailingTool("tool_primary", "unused"))
    logger = SpyTraceLogger()
    orchestrator = Orchestrator(
        router=DecisionRouter(
            RouteDecision(
                intent="alpha",
                confidence=0.9,
                secondary_intent=None,
                rationale="failure",
                routed_tool="tool_primary",
            )
        ),
        tool_registry=registry,
        trace_logger=logger,
    )

    with pytest.raises(ToolError):
        orchestrator.run(_make_input())

    assert logger.events == ["start", "route", "hook", "tool", "final", "end"]
    assert logger.hook_outcomes == [(None, None)]
    assert logger.final_decisions == [("alpha", 0.9)]
