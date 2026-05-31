"""Single-run runtime entry point."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from .router import ConfidenceHook, HookOutcome, RouteDecision, Router
from .tool import ToolRegistry
from .types import AudioInput, ImageInput

if TYPE_CHECKING:
    from .trace import TraceLogger


@dataclass
class RuntimeInput:
    """Normalized single-run input."""

    text: str | None
    image: ImageInput | None
    audio: AudioInput | None
    session_id: str


@dataclass
class RuntimeOutput:
    """Normalized single-run output."""

    text: str | None
    last_tool_output: dict | None
    route_decision: RouteDecision


class ToolArgBuilder(Protocol):
    """Build arguments for one tool invocation."""

    def build(self, tool_name: str, input: RuntimeInput, decision: RouteDecision) -> dict:
        """Map runtime state into one tool invocation payload."""


class DefaultToolArgBuilder:
    """Default runtime-to-tool argument mapping."""

    def build(self, tool_name: str, input: RuntimeInput, decision: RouteDecision) -> dict:
        """Map runtime input into one single tool invocation payload."""

        del decision
        args: dict[str, object] = {}
        if input.image is not None:
            args["image"] = input.image
        if input.audio is not None:
            args["audio"] = input.audio
        if input.text is not None:
            if tool_name == "describe_scene":
                args["query"] = input.text
            elif tool_name == "read_text":
                args["instruction"] = input.text
            else:
                args["text"] = input.text
        return args


class Orchestrator:
    """Run one route, one optional hook, and one optional tool invocation."""

    def __init__(
        self,
        router: Router,
        tool_registry: ToolRegistry,
        confidence_hook: ConfidenceHook | None = None,
        trace_logger: TraceLogger | None = None,
        tool_arg_builder: ToolArgBuilder | None = None,
        backends: list[object] | None = None,
    ) -> None:
        self.router = router
        self.tool_registry = tool_registry
        self.confidence_hook = confidence_hook
        self.trace_logger = trace_logger
        self.tool_arg_builder = tool_arg_builder or DefaultToolArgBuilder()
        self.backends = backends or []

    def run(self, input: RuntimeInput) -> RuntimeOutput:
        """Execute the minimal Phase 1 runtime loop."""

        if self.trace_logger is not None:
            self.trace_logger.on_run_start(input)

        decision = self.router.route(input)
        if self.trace_logger is not None:
            self.trace_logger.on_route(decision)

        hook = self.confidence_hook or getattr(self.router, "hook", None)
        if hook is not None:
            outcome = hook.on_route(decision, input)
            if self.trace_logger is not None:
                self.trace_logger.on_hook_outcome(outcome)
        else:
            outcome = HookOutcome(action="pass", clarify_prompt=None, fallback_tool=None)
            if self.trace_logger is not None:
                self.trace_logger.on_hook_outcome(None)

        if outcome.action == "clarify":
            output = RuntimeOutput(
                text=outcome.clarify_prompt,
                last_tool_output=None,
                route_decision=decision,
            )
            if self.trace_logger is not None:
                self.trace_logger.on_final_decision(None, None)
            self._finish(output)
            return output

        tool_name = decision.routed_tool
        if outcome.action == "fallback":
            tool_name = outcome.fallback_tool

        if tool_name is None:
            output = RuntimeOutput(text=None, last_tool_output=None, route_decision=decision)
            if self.trace_logger is not None:
                self.trace_logger.on_final_decision(decision.intent, decision.confidence)
            self._finish(output)
            return output

        args = self.tool_arg_builder.build(tool_name, input, decision)
        try:
            tool_output = self.tool_registry.invoke(tool_name, args)
        except Exception as exc:
            if self.trace_logger is not None:
                self.trace_logger.on_tool_invoke(tool_name, args, None, exc)
            output = RuntimeOutput(text=None, last_tool_output=None, route_decision=decision)
            if self.trace_logger is not None:
                self.trace_logger.on_final_decision(decision.intent, decision.confidence)
            self._finish(output)
            raise

        if self.trace_logger is not None:
            self.trace_logger.on_tool_invoke(tool_name, args, tool_output, None)

        final_text = tool_output.get("text") if isinstance(tool_output, dict) else None
        output = RuntimeOutput(
            text=final_text if isinstance(final_text, str) else None,
            last_tool_output=tool_output,
            route_decision=decision,
        )
        if self.trace_logger is not None:
            self.trace_logger.on_final_decision(decision.intent, decision.confidence)
        self._finish(output)
        return output

    def _finish(self, output: RuntimeOutput) -> None:
        if self.trace_logger is not None:
            self.trace_logger.on_run_end(output)
