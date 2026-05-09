"""Tool protocol, registry, and built-in implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import jsonschema

from ..backend import VLMBackend
from ..errors import ToolError, ValidationError
from ..types import AudioInput, ImageInput


class Tool(Protocol):
    """Executable tool with JSON-schema validated scalar arguments."""

    name: str
    input_schema: dict
    output_schema: dict

    def invoke(self, args: dict) -> dict:
        """Execute the tool with normalized arguments."""


class ToolRegistry:
    """Registry and JSON-schema gate for tools."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register one tool by immutable unique name."""

        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        """Return a registered tool or raise ToolError."""

        try:
            return self._tools[name]
        except KeyError as exc:
            raise ToolError(f"Unknown tool: {name}") from exc

    def invoke(self, name: str, args: dict) -> dict:
        """Validate scalar arguments, execute the tool, and validate output."""

        tool = self.get(name)
        scalar_args = _strip_runtime_objects(args)
        try:
            jsonschema.validate(instance=scalar_args, schema=tool.input_schema)
        except jsonschema.ValidationError as exc:
            raise ValidationError(f"Input schema validation failed for {name}.") from exc

        try:
            output = tool.invoke(args)
        except (ToolError, ValidationError):
            raise
        except Exception as exc:  # pragma: no cover - defensive wrapper
            raise ToolError(f"Tool execution failed for {name}.") from exc

        try:
            jsonschema.validate(instance=output, schema=tool.output_schema)
        except jsonschema.ValidationError as exc:
            raise ValidationError(f"Output schema validation failed for {name}.") from exc
        return output

    def list_tools(self) -> list[str]:
        """Return registered tool names sorted for deterministic tests."""

        return sorted(self._tools)


def _strip_runtime_objects(value: object) -> object:
    """Remove non-JSON runtime containers before schema validation."""

    if isinstance(value, (ImageInput, AudioInput)):
        return _SKIP
    if isinstance(value, dict):
        result: dict[str, object] = {}
        for key, item in value.items():
            cleaned = _strip_runtime_objects(item)
            if cleaned is not _SKIP:
                result[key] = cleaned
        return result
    if isinstance(value, list):
        result_list: list[object] = []
        for item in value:
            cleaned = _strip_runtime_objects(item)
            if cleaned is not _SKIP:
                result_list.append(cleaned)
        return result_list
    return value


_SKIP = object()


@dataclass
class DescribeSceneTool:
    """Built-in image-to-text helper."""

    vlm_backend: VLMBackend

    name = "describe_scene"
    input_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
        },
        "additionalProperties": False,
    }
    output_schema = {
        "type": "object",
        "properties": {
            "text": {"type": "string"},
        },
        "required": ["text"],
        "additionalProperties": False,
    }

    def invoke(self, args: dict) -> dict:
        """Generate a short scene description."""

        image = args.get("image")
        if not isinstance(image, ImageInput):
            raise ValidationError("describe_scene requires an ImageInput.")
        query = args.get("query")
        if query is not None and not isinstance(query, str):
            raise ValidationError("describe_scene.query must be a string.")
        return {"text": self.vlm_backend.generate(image=image, query=query or "")}


@dataclass
class ReadTextTool:
    """Built-in text extraction helper."""

    vlm_backend: VLMBackend

    name = "read_text"
    input_schema = {
        "type": "object",
        "properties": {
            "instruction": {"type": "string"},
        },
        "additionalProperties": False,
    }
    output_schema = {
        "type": "object",
        "properties": {
            "text": {"type": "string"},
        },
        "required": ["text"],
        "additionalProperties": False,
    }

    def invoke(self, args: dict) -> dict:
        """Generate text content from an image."""

        image = args.get("image")
        if not isinstance(image, ImageInput):
            raise ValidationError("read_text requires an ImageInput.")
        instruction = args.get("instruction")
        if instruction is not None and not isinstance(instruction, str):
            raise ValidationError("read_text.instruction must be a string.")
        return {"text": self.vlm_backend.generate(image=image, query=instruction or "")}
