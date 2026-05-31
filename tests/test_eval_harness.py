from __future__ import annotations

import json

from edgent_core import (
    ClassifierRouter,
    EvalHarness,
    IntentNode,
    JsonlTraceLogger,
    Orchestrator,
    RuntimeOutput,
    StubLLMBackend,
    TaxonomySchema,
    ToolRegistry,
)


class RequestTool:
    name = "tool_request"
    input_schema = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "additionalProperties": False,
    }
    output_schema = {
        "type": "object",
        "properties": {"text": {"type": "string"}, "payload": {"type": "object"}},
        "required": ["text", "payload"],
        "additionalProperties": False,
    }

    def invoke(self, args: dict) -> dict:
        del args
        return {"text": "request output", "payload": {"ok": True}}


class PlainTool:
    name = "tool_plain"
    input_schema = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "additionalProperties": False,
    }
    output_schema = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
        "additionalProperties": False,
    }

    def invoke(self, args: dict) -> dict:
        del args
        return {"text": "plain output"}


class SimplePayloadValidator:
    def validate(self, entry, output: RuntimeOutput) -> bool | None:  # noqa: ANN001
        del entry
        if output.last_tool_output is None:
            return False
        return "payload" in output.last_tool_output


def _make_orchestrator(tmp_path) -> Orchestrator:  # noqa: ANN001
    taxonomy = TaxonomySchema(
        primary_nodes=[
            IntentNode(id="alpha", label="Alpha", parent_id=None, routed_tool="tool_plain"),
            IntentNode(id="beta", label="Beta", parent_id=None, routed_tool="tool_request"),
        ],
        secondary_nodes_by_parent={},
        tool_suggestions_by_intent={"alpha": "tool_plain", "beta": "tool_request"},
        clarify_threshold=0.75,
        fallback_threshold=0.55,
    )
    registry = ToolRegistry()
    registry.register(PlainTool())
    registry.register(RequestTool())
    return Orchestrator(
        router=ClassifierRouter(taxonomy=taxonomy, llm_backend=StubLLMBackend()),
        tool_registry=registry,
        trace_logger=JsonlTraceLogger(tmp_path / "traces"),
    )


def _write_dataset(path) -> None:  # noqa: ANN001
    entries = [
        {
            "case_id": "case-1",
            "text": "route alpha",
            "image_ref": None,
            "audio_ref": None,
            "expected_primary_intent": "alpha",
            "expected_secondary_intent": None,
            "expected_request_intent": None,
            "expected_slots": None,
            "notes": None,
        },
        {
            "case_id": "case-2",
            "text": "route beta",
            "image_ref": None,
            "audio_ref": None,
            "expected_primary_intent": "beta",
            "expected_secondary_intent": None,
            "expected_request_intent": "request",
            "expected_slots": None,
            "notes": None,
        },
    ]
    with path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry))
            handle.write("\n")


def test_eval_harness_without_payload_validator(tmp_path) -> None:  # noqa: ANN001
    dataset_path = tmp_path / "entries.jsonl"
    _write_dataset(dataset_path)
    harness = EvalHarness()
    dataset = harness.load_dataset(str(dataset_path))

    result = harness.run(dataset, _make_orchestrator(tmp_path))
    summary = harness.report(result)

    assert len(result.case_results) == 2
    assert all(case.schema_validation_passed is None for case in result.case_results)
    assert summary.primary_intent_accuracy == 1.0
    assert summary.schema_validation_pass_rate == 0.0
    assert summary.latency_p50_ms >= 0.0
    assert summary.latency_p95_ms >= 0.0
    assert summary.cost_total_usd == 0.0


def test_eval_harness_with_payload_validator(tmp_path) -> None:  # noqa: ANN001
    dataset_path = tmp_path / "entries.jsonl"
    _write_dataset(dataset_path)
    harness = EvalHarness(payload_validator=SimplePayloadValidator())
    dataset = harness.load_dataset(str(dataset_path))

    result = harness.run(dataset, _make_orchestrator(tmp_path))
    summary = harness.report(result)

    assert result.case_results[0].schema_validation_passed is None
    assert result.case_results[1].schema_validation_passed is True
    assert summary.schema_validation_pass_rate == 1.0
