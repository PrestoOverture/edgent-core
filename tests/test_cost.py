"""Tests for Usage, CostCalculator, and cost tracking in trace."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from edgent_core.backend.base import StubLLMBackend, StubVLMBackend
from edgent_core.orchestrator import Orchestrator, RuntimeInput
from edgent_core.router import RouteDecision
from edgent_core.tool import ToolRegistry
from edgent_core.trace import JsonlTraceLogger
from edgent_core.types import CostCalculator, Usage


class _FakeCostCalculator:
    """Test calculator: $0.001 per 1000 prompt tokens, $0.002 per 1000 completion tokens."""

    def calculate(self, usage: Usage) -> float:
        return (usage.prompt_tokens * 0.001 + usage.completion_tokens * 0.002) / 1000.0


def test_usage_defaults_to_zero():
    u = Usage()
    assert u.prompt_tokens == 0
    assert u.completion_tokens == 0


def test_stub_backends_have_zero_usage():
    llm = StubLLMBackend()
    vlm = StubVLMBackend()
    assert llm.last_usage.prompt_tokens == 0
    assert llm.last_usage.completion_tokens == 0
    assert vlm.last_usage.prompt_tokens == 0
    assert vlm.last_usage.completion_tokens == 0


def test_trace_logger_cost_none_without_calculator():
    logger = JsonlTraceLogger()
    # Simulate a run
    logger._backends = []
    assert logger._compute_cost() is None


def test_trace_logger_cost_zero_with_calculator_no_backends():
    logger = JsonlTraceLogger(cost_calculator=_FakeCostCalculator(), backends=[])
    assert logger._compute_cost() == 0.0


def test_trace_logger_cost_computed_from_backend_usage():
    class _Backend:
        last_usage = Usage(prompt_tokens=1000, completion_tokens=500)

    logger = JsonlTraceLogger(
        cost_calculator=_FakeCostCalculator(),
        backends=[_Backend()],
    )
    # (1000 * 0.001 + 500 * 0.002) / 1000 = (1.0 + 1.0) / 1000 = 0.002
    assert logger._compute_cost() == 0.002


def test_trace_logger_cost_aggregates_multiple_backends():
    class _LLM:
        last_usage = Usage(prompt_tokens=500, completion_tokens=200)

    class _VLM:
        last_usage = Usage(prompt_tokens=2000, completion_tokens=300)

    logger = JsonlTraceLogger(
        cost_calculator=_FakeCostCalculator(),
        backends=[_LLM(), _VLM()],
    )
    # Total: prompt=2500, completion=500
    # (2500 * 0.001 + 500 * 0.002) / 1000 = (2.5 + 1.0) / 1000 = 0.0035
    assert logger._compute_cost() == 0.0035


def test_trace_logger_cost_ignores_backends_without_usage():
    class _NoUsage:
        name = "no-usage"

    class _WithUsage:
        last_usage = Usage(prompt_tokens=100, completion_tokens=50)

    logger = JsonlTraceLogger(
        cost_calculator=_FakeCostCalculator(),
        backends=[_NoUsage(), _WithUsage()],
    )
    # Only _WithUsage contributes: (100 * 0.001 + 50 * 0.002) / 1000 = 0.0002
    assert abs(logger._compute_cost() - 0.0002) < 1e-10


def test_trace_record_has_cost_usd_in_jsonl():
    """End-to-end: orchestrator writes trace with cost_usd when calculator is provided."""

    class _FakeRouter:
        def route(self, input: RuntimeInput) -> RouteDecision:
            return RouteDecision(
                intent="test", confidence=0.9, secondary_intent=None,
                rationale=None, routed_tool=None,
            )

    class _Backend:
        last_usage = Usage(prompt_tokens=800, completion_tokens=200)

    backend = _Backend()
    calc = _FakeCostCalculator()

    with tempfile.TemporaryDirectory() as tmpdir:
        logger = JsonlTraceLogger(Path(tmpdir), cost_calculator=calc, backends=[backend])
        orch = Orchestrator(
            router=_FakeRouter(),
            tool_registry=ToolRegistry(),
            trace_logger=logger,
            backends=[backend],
        )
        inp = RuntimeInput(text="hello", image=None, audio=None, session_id="s1")
        orch.run(inp)

        # Read the trace file
        trace_file = Path(tmpdir) / "s1.jsonl"
        assert trace_file.exists()
        record = json.loads(trace_file.read_text().strip().split("\n")[-1])
        # cost_usd should be non-None and > 0
        assert record["cost_usd"] is not None
        assert record["cost_usd"] > 0
        # Expected: (800 * 0.001 + 200 * 0.002) / 1000 = 0.0012
        assert abs(record["cost_usd"] - 0.0012) < 1e-10


def test_trace_record_cost_usd_none_without_calculator():
    """Without a calculator, cost_usd remains null."""

    class _FakeRouter:
        def route(self, input: RuntimeInput) -> RouteDecision:
            return RouteDecision(
                intent="test", confidence=0.9, secondary_intent=None,
                rationale=None, routed_tool=None,
            )

    with tempfile.TemporaryDirectory() as tmpdir:
        logger = JsonlTraceLogger(Path(tmpdir))
        orch = Orchestrator(
            router=_FakeRouter(),
            tool_registry=ToolRegistry(),
            trace_logger=logger,
        )
        inp = RuntimeInput(text="hello", image=None, audio=None, session_id="s2")
        orch.run(inp)

        trace_file = Path(tmpdir) / "s2.jsonl"
        record = json.loads(trace_file.read_text().strip().split("\n")[-1])
        assert record["cost_usd"] is None


def test_cost_calculator_protocol_is_runtime_checkable():
    """CostCalculator is a runtime-checkable protocol."""
    calc = _FakeCostCalculator()
    assert isinstance(calc, CostCalculator)
