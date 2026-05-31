"""Dataset and evaluation primitives."""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ..orchestrator import Orchestrator, RuntimeInput, RuntimeOutput
from ..trace import JsonlTraceLogger, TraceRecord
from ..types import AudioInput, ImageInput


@dataclass
class EvalEntry:
    """One evaluation case."""

    case_id: str
    text: str | None
    image_ref: str | None
    audio_ref: str | None
    expected_primary_intent: str
    expected_secondary_intent: str | None
    expected_request_intent: str | None
    expected_slots: dict | None
    notes: str | None


@dataclass
class EvalDataset:
    """Materialized evaluation dataset."""

    schema_version: str
    entries: list[EvalEntry]


@dataclass
class EvalCaseResult:
    """Outcome for one evaluation case."""

    case_id: str
    trace_id: str
    matched_gold_primary_intent: bool
    matched_gold_secondary_intent: bool | None
    schema_validation_passed: bool | None


@dataclass
class EvalRunResult:
    """Full run output over one dataset."""

    case_results: list[EvalCaseResult]


@dataclass
class EvalSummary:
    """Minimal summary metrics for one run."""

    primary_intent_accuracy: float
    schema_validation_pass_rate: float
    latency_p50_ms: float
    latency_p95_ms: float
    cost_total_usd: float


class PayloadValidator(Protocol):
    """Optional adapter-supplied payload validator."""

    def validate(self, entry: EvalEntry, output: RuntimeOutput) -> bool | None:
        """Return validation status for one output or None when not applicable."""


class EvalHarness:
    """Load JSONL datasets, run them through one orchestrator, and summarize results."""

    def __init__(self, payload_validator: PayloadValidator | None = None) -> None:
        self.payload_validator = payload_validator
        self._last_trace_records: list[TraceRecord] = []

    def load_dataset(self, path: str) -> EvalDataset:
        """Load JSONL entries into an in-memory dataset."""

        entries: list[EvalEntry] = []
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                entries.append(EvalEntry(**payload))
        return EvalDataset(schema_version="1.0", entries=entries)

    def run(self, dataset: EvalDataset, orchestrator: Orchestrator) -> EvalRunResult:
        """Run every case once and preserve trace references."""

        self._last_trace_records = []
        local_temp_dir: tempfile.TemporaryDirectory[str] | None = None
        run_orchestrator = orchestrator
        if not isinstance(orchestrator.trace_logger, JsonlTraceLogger):
            local_temp_dir = tempfile.TemporaryDirectory()
            logger = JsonlTraceLogger(Path(local_temp_dir.name))
            run_orchestrator = Orchestrator(
                router=orchestrator.router,
                tool_registry=orchestrator.tool_registry,
                confidence_hook=orchestrator.confidence_hook,
                trace_logger=logger,
                tool_arg_builder=orchestrator.tool_arg_builder,
                backends=orchestrator.backends,
            )

        try:
            case_results: list[EvalCaseResult] = []
            for entry in dataset.entries:
                try:
                    runtime_input = RuntimeInput(
                        text=entry.text,
                        image=ImageInput(path=entry.image_ref) if entry.image_ref else None,
                        audio=AudioInput(path=entry.audio_ref) if entry.audio_ref else None,
                        session_id=f"eval-{entry.case_id}",
                    )
                    output = run_orchestrator.run(runtime_input)
                    trace_record = self._require_trace_record(run_orchestrator)
                    self._last_trace_records.append(trace_record)

                    schema_validation_passed: bool | None = None
                    if entry.expected_request_intent is not None and self.payload_validator is not None:
                        schema_validation_passed = self.payload_validator.validate(entry, output)

                    matched_secondary = None
                    if entry.expected_secondary_intent is not None:
                        matched_secondary = (
                            output.route_decision.secondary_intent == entry.expected_secondary_intent
                        )

                    case_results.append(
                        EvalCaseResult(
                            case_id=entry.case_id,
                            trace_id=trace_record.trace_id,
                            matched_gold_primary_intent=(
                                output.route_decision.intent == entry.expected_primary_intent
                            ),
                            matched_gold_secondary_intent=matched_secondary,
                            schema_validation_passed=schema_validation_passed,
                        )
                    )
                except Exception as exc:
                    # Log failure but continue with remaining cases
                    import sys
                    print(f"  WARN: case {entry.case_id} failed: {exc}", file=sys.stderr)
                    case_results.append(
                        EvalCaseResult(
                            case_id=entry.case_id,
                            trace_id="error",
                            matched_gold_primary_intent=False,
                            matched_gold_secondary_intent=None,
                            schema_validation_passed=None,
                        )
                    )
            return EvalRunResult(case_results=case_results)
        finally:
            if local_temp_dir is not None:
                local_temp_dir.cleanup()

    def report(self, result: EvalRunResult) -> EvalSummary:
        """Compute the minimal Phase 1 summary from the last run."""

        total = len(result.case_results)
        primary_hits = sum(1 for case in result.case_results if case.matched_gold_primary_intent)
        primary_intent_accuracy = (primary_hits / total) if total else 0.0

        validation_values = [
            case.schema_validation_passed
            for case in result.case_results
            if case.schema_validation_passed is not None
        ]
        if validation_values:
            validation_hits = sum(1 for value in validation_values if value)
            schema_validation_pass_rate = validation_hits / len(validation_values)
        else:
            schema_validation_pass_rate = 0.0

        latencies = sorted(
            record.latency_ms for record in self._last_trace_records if record.latency_ms is not None
        )
        costs = [record.cost_usd for record in self._last_trace_records if record.cost_usd is not None]

        return EvalSummary(
            primary_intent_accuracy=primary_intent_accuracy,
            schema_validation_pass_rate=schema_validation_pass_rate,
            latency_p50_ms=_percentile(latencies, 50.0),
            latency_p95_ms=_percentile(latencies, 95.0),
            cost_total_usd=sum(costs) if costs else 0.0,
        )

    def _require_trace_record(self, orchestrator: Orchestrator) -> TraceRecord:
        logger = orchestrator.trace_logger
        if not isinstance(logger, JsonlTraceLogger) or logger.last_record is None:
            raise RuntimeError("EvalHarness requires a JsonlTraceLogger-backed orchestrator.")
        return logger.last_record


def _percentile(values: list[int], percentile: float) -> float:
    """Compute a simple linear percentile for a sorted integer list."""

    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    index = (len(values) - 1) * (percentile / 100.0)
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)
    weight = index - lower
    return float(values[lower] + (values[upper] - values[lower]) * weight)
