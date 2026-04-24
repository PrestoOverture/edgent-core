from __future__ import annotations

import json

from edgent_core import HookOutcome, JsonlTraceLogger, RouteDecision, RuntimeInput, RuntimeOutput


def test_jsonl_trace_logger_writes_valid_record(tmp_path) -> None:  # noqa: ANN001
    logger = JsonlTraceLogger(tmp_path / "trace.jsonl")
    input_data = RuntimeInput(text="input", image=None, audio=None, session_id="session")
    output = RuntimeOutput(
        text="output",
        last_tool_output={"text": "output"},
        route_decision=RouteDecision(
            intent="alpha",
            confidence=0.9,
            secondary_intent=None,
            rationale="trace",
            routed_tool="tool_alpha",
        ),
    )

    logger.on_run_start(input_data)
    logger.on_route(output.route_decision)
    logger.on_hook_outcome(HookOutcome(action="pass", clarify_prompt=None, fallback_tool=None))
    logger.on_tool_invoke("tool_alpha", {"text": "input"}, {"text": "output"}, None)
    logger.on_final_decision("alpha", 0.9)
    logger.on_run_end(output)

    payload = json.loads((tmp_path / "trace.jsonl").read_text(encoding="utf-8").strip())
    assert payload["trace_id"]
    assert payload["session_id"] == "session"
    assert payload["route_decision"]["intent"] == "alpha"
    assert payload["hook_action"] == "pass"
    assert payload["fallback_tool"] is None
    assert payload["tool_calls"][0]["tool_name"] == "tool_alpha"
    assert payload["final_intent"] == "alpha"
    assert payload["final_confidence"] == 0.9
    assert payload["final_text"] == "output"
