# edgent-core

A minimal agent runtime kernel for intent routing, tool orchestration, and evaluation.

## Install

```bash
pip install edgent-core
```

From source:

```bash
git clone https://github.com/PrestoOverture/edgent-core.git
cd edgent-core
pip install -e .
```

## Quick start

```python
from edgent_core import (
    ClassifierRouter,
    IntentNode,
    Orchestrator,
    StubLLMBackend,
    ToolRegistry,
    TaxonomySchema,
)

# Define a taxonomy
taxonomy = TaxonomySchema(
    primary_nodes=[
        IntentNode(id="greet", label="Greet", parent_id=None, routed_tool=None),
        IntentNode(id="help", label="Help", parent_id=None, routed_tool=None),
    ],
    secondary_nodes_by_parent={},
    tool_suggestions_by_intent={},
    clarify_threshold=0.75,
    fallback_threshold=0.55,
)

# Build orchestrator
orch = Orchestrator(
    router=ClassifierRouter(taxonomy=taxonomy, llm_backend=StubLLMBackend()),
    tool_registry=ToolRegistry(),
)

# Run
from edgent_core import RuntimeInput
output = orch.run(RuntimeInput(text="hello there"))
print(output.route_decision.intent)  # "greet"
```

## Public API

All public symbols are exported from `edgent_core.__init__` via `__all__`.

| Layer | Symbols |
|-------|---------|
| **Types** | `ImageInput`, `AudioInput`, `VisualCue`, `Usage`, `CostCalculator` |
| **Errors** | `CoreRuntimeError`, `BackendError`, `RouteError`, `ToolError`, `ValidationError` |
| **Router** | `Router`, `ClassifierRouter`, `FallbackChain`, `SupervisorRouter`, `RouteDecision`, `TaxonomySchema`, `IntentNode`, `ConfidenceHook`, `HookOutcome`, `PreRouteHook`, `VisualCueProvider`, `FallbackTrigger` |
| **Tool** | `Tool`, `ToolRegistry`, `DescribeSceneTool`, `ReadTextTool` |
| **Backend** | `LLMBackend`, `VLMBackend`, `OpenAILLMBackend`, `OpenAIVLMBackend`, `StubLLMBackend`, `StubVLMBackend`, `ClassifierOutput` |
| **Orchestrator** | `Orchestrator`, `RuntimeInput`, `RuntimeOutput`, `ToolArgBuilder`, `DefaultToolArgBuilder` |
| **Trace** | `TraceLogger`, `JsonlTraceLogger`, `TraceRecord`, `ToolCallTrace` |
| **Eval** | `EvalHarness`, `EvalEntry`, `EvalDataset`, `EvalRunResult`, `EvalCaseResult`, `EvalSummary`, `PayloadValidator` |

## Development

```bash
pip install -e ".[dev]"
ruff check edgent_core/ tests/
mypy edgent_core/ --ignore-missing-imports
pytest tests/
```

## License

MIT
