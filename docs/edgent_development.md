# edgent-core — 开发操作手册

通用 Agent runtime kernel。本文件只记录 **当前 repo 内事实**（stack、命令、已实现模块、约束、限制）。跨 repo 契约见 workspace `docs/interfaces.md`。

## 阶段状态

- Phase 4 Part D 完成（2026-05-31）：CI 门禁、超时/重试、并发 eval、开源就绪
- 本地 `pytest` **47 passed**
- CI：`.github/workflows/ci.yml`（ruff → mypy → pytest，push/PR 触发）
- LICENSE：MIT

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
pytest
```

依赖：

- `jsonschema`
- `openai`
- `pytest` / `ruff` / `mypy`（optional-dev）

Python ≥ 3.10（使用 `|` union 语法）。

## 目录结构

```text
edgent-core/
  edgent_core/              # 包根，所有 stable 符号在 __init__.py 集中导出
    backend/                # VLM / LLM Protocol + Stub backend + OpenAI-compatible backend
    eval/                   # EvalHarness / EvalEntry / EvalDataset / PayloadValidator Protocol
    router/                 # Router / ClassifierRouter / FallbackChain / ConfidenceHook / TaxonomySchema / IntentNode / SupervisorRouter stub
    tool/                   # Tool Protocol / ToolRegistry / DescribeSceneTool / ReadTextTool
    errors.py               # 异常族：CoreRuntimeError / RouteError / ToolError / BackendError / ValidationError
    orchestrator.py         # Orchestrator / RuntimeInput / RuntimeOutput / ToolArgBuilder（单轮编排）
    trace.py                # TraceLogger Protocol（6 钩子）/ JsonlTraceLogger / TraceRecord / ToolCallTrace
    types.py                # ImageInput / AudioInput 抽象输入容器
  tests/                    # 契约自测
  docs/
  pyproject.toml
  README.md
```

## 已实现模块

- `Router` Protocol / `RouteDecision`
- `ClassifierRouter` 单次闭集分类（backend 可为 `StubLLMBackend` 或 `OpenAILLMBackend`）
- `FallbackChain` + `FallbackTrigger`：异常 / 低置信 / 超时三类降级条件
- `ConfidenceHook` Protocol / `HookOutcome`
- `TaxonomySchema` / `IntentNode`（契约层，不含业务枚举）
- `SupervisorRouter`：`NotImplementedError` stub（§4.11 experimental）
- `Tool` Protocol / `ToolRegistry`：register / get / list / invoke（含 `input_schema` / `output_schema` JSON Schema 校验）
- `DescribeSceneTool` / `ReadTextTool`：最小实现，output 只返回 `{"text": str}`
- `VLMBackend` / `LLMBackend` Protocol + `ClassifierOutput` dataclass
- `StubVLMBackend` / `StubLLMBackend`：deterministic，不接外部 API
- `OpenAIVLMBackend`：OpenAI-compatible vision chat completion backend
- `OpenAILLMBackend`：OpenAI-compatible text chat completion backend；默认尝试 JSON response format，失败时回落到严格 JSON prompt retry
- `ToolArgBuilder` Protocol / `DefaultToolArgBuilder`：runtime input 到 tool args 的通用映射，可由 adapter 注入自定义实现
- `Orchestrator.run()` 单轮编排：`on_run_start` → `route` → `on_route` → hook → `on_hook_outcome` → 最多一次 tool → `on_tool_invoke` → `on_final_decision` → `on_run_end`
- `RuntimeInput` / `RuntimeOutput`（后者含 `last_tool_output`）
- `TraceLogger` Protocol 六钩子 + `JsonlTraceLogger`
- `TraceRecord` / `ToolCallTrace`
- `EvalHarness`：`load_dataset` / `run` / `report`
- `EvalEntry` / `EvalDataset` / `EvalCaseResult` / `EvalRunResult` / `EvalSummary`
- `PayloadValidator` Protocol（可选注入）

## Stub backend

- `StubLLMBackend`：确定性闭集分类与文本生成，不接外部 API
- `StubVLMBackend`：确定性 image-to-text 返回值，不接外部 API
- 两者用于本地契约测试与 skeleton 自测，不代表生产策略

## OpenAI-compatible backend

- `OpenAIVLMBackend`：调用 `chat.completions.create()`，发送文本 query + image data URL，返回文本内容；`request_timeout=60s` / `max_retries=2`
- `OpenAILLMBackend`：调用 `chat.completions.create()` 做闭集分类，返回 `ClassifierOutput(choice, confidence)`；`request_timeout=30s` / `max_retries=2`；logprobs 可选
- 两者只实现通用 provider adapter，不写任何 adapter-specific prompt 或 taxonomy
- 超时/连接错误通过 `_call_api()` 包装为 `BackendError`，不被 classify 的 retry 逻辑吞掉

## Trace Log

- 默认落盘路径：`./traces/{session_id}.jsonl`
- `JsonlTraceLogger(path=...)` 可覆盖默认路径；传入目录时按 session 拆文件，传入 `.jsonl` 文件时直接写该文件
- `JsonlTraceLogger` 实现 `TraceLogger` Protocol 全部 6 钩子
- `args_digest` / `output_digest` 策略：`repr(payload)` 后做 `sha256`，取前 16 个十六进制字符
- `cost_usd` 当前默认写 `null`（未接入成本采集）
- `HookOutcome` 通过 `TYPE_CHECKING` 块 import，避免 `trace.py` / `router/` 循环依赖

## Eval Harness 当前行为

- `load_dataset(path)` 读 jsonl，当前固定返回 `schema_version = "1.0"`
- `run(dataset, orchestrator, max_workers=1)`：`max_workers=1` 串行（默认），>1 并发（`ThreadPoolExecutor`，每 worker 独立 Orchestrator clone + 独立 trace logger）
- 若注入 `PayloadValidator`，每条 case 调用一次 `validator.validate(entry, output)` 产出 `schema_validation_passed`
- `report(result)` 计算指标：
  - `primary_intent_accuracy`
  - `schema_validation_pass_rate`（全部 `schema_validation_passed = None` 时按 `0.0` 处理）
  - `latency_p50_ms` / `latency_p95_ms`
  - `cost_total_usd`
- 不计算 secondary-intent accuracy；原始布尔值保留在 `EvalCaseResult.matched_gold_secondary_intent`

## 当前限制 / 未覆盖范围

- `ClassifierRouter` 仅支持单次闭集分类路径，不做多 hop
- `Orchestrator.run()` 每轮最多一次 tool 调用；无多 tool 循环 / stream / subagent
- Trace 落盘未接入轮转 / 压缩
- Eval 指标不含 `confidence_calibration_error` 等派生指标（原始字段已保留在 `TraceRecord`，公式留后）
- 真实 backend 的速率限制与 provider-specific telemetry 仍未系统化

## 关键约束

- 不引入 graph runtime 依赖（LangGraph / LangChain 等）
- 对外 API 先对齐 workspace `docs/interfaces.md` §4，再调整实现
- 包代码 / 文档不得出现 adapter-specific 业务词汇；core 只保留 generic runtime / backend / tool / eval 语义
