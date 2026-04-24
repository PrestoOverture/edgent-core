from __future__ import annotations

import pytest

from edgent_core import ToolError, ToolRegistry, ValidationError


class EchoTool:
    name = "echo"
    input_schema = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
        "additionalProperties": False,
    }
    output_schema = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
        "additionalProperties": False,
    }

    def invoke(self, args: dict) -> dict:
        return {"text": args["value"]}


def test_register_get_and_invoke() -> None:
    registry = ToolRegistry()
    registry.register(EchoTool())

    assert registry.get("echo").name == "echo"
    assert registry.list_tools() == ["echo"]
    assert registry.invoke("echo", {"value": "hello"}) == {"text": "hello"}


def test_get_unregistered_tool_raises() -> None:
    registry = ToolRegistry()

    with pytest.raises(ToolError):
        registry.get("missing")


def test_invoke_validation_error() -> None:
    registry = ToolRegistry()
    registry.register(EchoTool())

    with pytest.raises(ValidationError):
        registry.invoke("echo", {"value": 3})
