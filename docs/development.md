# edgent-core — 开发操作手册

通用 Agent 运行时，候选开源。Router 原语 + Orchestrator + Tool Registry + Hook + Trace + Eval harness，**不含任何盲人业务**。

本文件是这个 repo 的事实说明（stack、命令、约束）。跨 repo 信息见 workspace `docs/`。

> 本 repo 在 Week 3 Phase 1 开始落地，当前骨架阶段。

## Environment

TBD — 独立 venv：

```bash
python -m venv .venv
source .venv/bin/activate  # Python 3.12.x
pip install -e .[dev]
```

预期依赖：`pydantic` / `openai` / `dashscope` / `langgraph`（TBD）/ `pytest` / `ruff` / `mypy`

## Repo 职责

- `Router` 接口 + `RouteDecision` 结构
- `ClassifierRouter`（reference impl，闭集分类 + 单次 LLM 调用）
- `SupervisorRouter`（reference impl，master + 动态 subagent，V1 不使用但保留接口）
- `ConfidenceHook` / `FallbackChain` 通用挂载点
- Tool Registry + Tool schema 校验
- VLM / LLM Backend 抽象（≥2 个 VLM、≥2 个 LLM 可切换）
- Trace Log 统一 schema
- Eval Harness（四指标 pipeline + 三项 Agent 特化指标 + dashboard）
- Graceful Degradation 3-tier 通用框架
- `examples/`（通用 demo，不含盲人业务）

## 关键约束

- **零盲人业务**：任何涉及"读字 / 导航 / 紧急求助 / 菜单 / 招牌"的概念都不能出现在本 repo
- 所有对外 API 保持可开源状态（license headers、依赖审查）
- 接口变更必须先更新 workspace `docs/interfaces.md` 再改本 repo
- `ClassifierRouter` 低延迟单次 LLM 调用是硬约束（V1 依赖，不能退化成多 hop）

## 预期目录

```
edgent-core/
  edgent_core/
    router/         ← Router 原语 + 2 个 reference impl
    orchestrator/
    tools/          ← Tool Registry + built-in
    backends/       ← VLM / LLM 抽象
    hooks/          ← ConfidenceHook / FallbackChain
    trace/
    eval/
  examples/         ← 通用 demo（如 travel）
  tests/
  docs/
  pyproject.toml
```
