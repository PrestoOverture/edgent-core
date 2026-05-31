"""Trace protocol and JSONL logger."""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol

from .router import RouteDecision
from .types import CostCalculator, Usage

if TYPE_CHECKING:
    from .orchestrator import RuntimeInput, RuntimeOutput
    from .router import HookOutcome


@dataclass
class ToolCallTrace:
    """One tool invocation trace entry."""

    tool_name: str
    args_digest: str | None
    output_digest: str | None
    error: str | None


@dataclass
class TraceRecord:
    """One persisted runtime trace row."""

    trace_id: str
    session_id: str
    timestamp: str
    text: str | None
    image_ref: str | None
    audio_ref: str | None
    route_decision: RouteDecision
    hook_action: Literal["pass", "clarify", "fallback"] | None
    fallback_tool: str | None
    tool_calls: list[ToolCallTrace]
    final_text: str | None
    final_intent: str | None
    final_confidence: float | None
    latency_ms: int | None
    cost_usd: float | None


class TraceLogger(Protocol):
    """Trace hook interface."""

    def on_run_start(self, input: RuntimeInput) -> None:
        """Record the start of one orchestrator run."""

    def on_route(self, decision: RouteDecision) -> None:
        """Record the router decision."""

    def on_hook_outcome(self, outcome: HookOutcome | None) -> None:
        """Record the confidence-hook outcome for the current run."""

    def on_tool_invoke(
        self,
        tool_name: str,
        args: dict,
        output: dict | None,
        error: Exception | None,
    ) -> None:
        """Record one tool call event."""

    def on_final_decision(self, intent: str | None, confidence: float | None) -> None:
        """Record the final adopted intent and confidence."""

    def on_run_end(self, output: RuntimeOutput) -> None:
        """Flush one completed trace record."""


class JsonlTraceLogger:
    """Write one JSON line per completed run."""

    def __init__(
        self,
        path: str | Path | None = None,
        cost_calculator: CostCalculator | None = None,
        backends: list[object] | None = None,
    ) -> None:
        self._configured_path = Path(path) if path is not None else None
        self._cost_calculator = cost_calculator
        self._backends = backends or []
        self.last_record: TraceRecord | None = None
        self._reset()

    def on_run_start(self, input: RuntimeInput) -> None:
        """Initialize state for one run."""

        self._reset()
        self._trace_id: str | None = str(uuid.uuid4())
        self._session_id: str | None = input.session_id
        self._timestamp: str | None = datetime.now(timezone.utc).isoformat()
        self._text: str | None = input.text
        self._image_ref: str | None = input.image.persistent_ref() if input.image is not None else None
        self._audio_ref: str | None = input.audio.persistent_ref() if input.audio is not None else None
        self._started_at: float | None = time.monotonic()
        self._target_path = self._resolve_path(input.session_id)

    def on_route(self, decision: RouteDecision) -> None:
        """Store the route decision for the active run."""

        self._route_decision: RouteDecision | None = decision

    def on_tool_invoke(
        self,
        tool_name: str,
        args: dict,
        output: dict | None,
        error: Exception | None,
    ) -> None:
        """Append one tool call record to the current run."""

        self._tool_calls.append(
            ToolCallTrace(
                tool_name=tool_name,
                args_digest=_digest(args),
                output_digest=_digest(output),
                error=str(error) if error is not None else None,
            )
        )

    def on_hook_outcome(self, outcome: HookOutcome | None) -> None:
        """Capture hook output for the active run."""

        if outcome is None:
            self._hook_action = None
            self._fallback_tool = None
            return
        self._hook_action = outcome.action
        self._fallback_tool = outcome.fallback_tool

    def on_final_decision(self, intent: str | None, confidence: float | None) -> None:
        """Capture the final adopted intent for the active run."""

        self._final_decision_recorded = True
        self._final_intent = intent
        self._final_confidence = confidence

    def on_run_end(self, output: RuntimeOutput) -> None:
        """Write the completed record to JSONL storage."""

        route_decision = self._route_decision or output.route_decision
        latency_ms = None
        if self._started_at is not None:
            latency_ms = int((time.monotonic() - self._started_at) * 1000)

        if self._final_decision_recorded:
            final_intent = self._final_intent
            final_confidence = self._final_confidence
        else:
            final_intent = output.route_decision.intent
            final_confidence = output.route_decision.confidence

        cost_usd = self._compute_cost()

        record = TraceRecord(
            trace_id=self._trace_id or str(uuid.uuid4()),
            session_id=self._session_id or "",
            timestamp=self._timestamp or datetime.now(timezone.utc).isoformat(),
            text=self._text,
            image_ref=self._image_ref,
            audio_ref=self._audio_ref,
            route_decision=route_decision,
            hook_action=self._hook_action,
            fallback_tool=self._fallback_tool,
            tool_calls=list(self._tool_calls),
            final_text=output.text,
            final_intent=final_intent,
            final_confidence=final_confidence,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
        )
        self._target_path.parent.mkdir(parents=True, exist_ok=True)
        with self._target_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(record), ensure_ascii=True))
            handle.write("\n")
        self.last_record = record

    def _compute_cost(self) -> float | None:
        """Aggregate token usage from backends and compute cost."""

        if self._cost_calculator is None:
            return None
        total_prompt = 0
        total_completion = 0
        for backend in self._backends:
            usage = getattr(backend, "last_usage", None)
            if isinstance(usage, Usage):
                total_prompt += usage.prompt_tokens
                total_completion += usage.completion_tokens
        if total_prompt == 0 and total_completion == 0:
            return 0.0
        return self._cost_calculator.calculate(
            Usage(prompt_tokens=total_prompt, completion_tokens=total_completion)
        )

    def _resolve_path(self, session_id: str) -> Path:
        if self._configured_path is None:
            return Path("traces") / f"{session_id}.jsonl"
        if self._configured_path.suffix == ".jsonl":
            return self._configured_path
        return self._configured_path / f"{session_id}.jsonl"

    def _reset(self) -> None:
        self._trace_id = None
        self._session_id = None
        self._timestamp = None
        self._text = None
        self._image_ref = None
        self._audio_ref = None
        self._route_decision = None
        self._hook_action = None
        self._fallback_tool = None
        self._tool_calls: list[ToolCallTrace] = []
        self._final_decision_recorded = False
        self._final_intent = None
        self._final_confidence = None
        self._started_at = None
        self._target_path = Path("traces") / "runtime.jsonl"


def _digest(value: object) -> str | None:
    """Create a short digest for arbitrarily large payloads."""

    if value is None:
        return None
    payload = repr(value).encode("utf-8", errors="replace")
    return hashlib.sha256(payload).hexdigest()[:16]
