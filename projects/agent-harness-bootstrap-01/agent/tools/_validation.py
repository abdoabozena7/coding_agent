"""Small, strict validator for the JSON-schema subset used by tools."""

from __future__ import annotations

import math
from typing import Any


class ToolArgumentError(ValueError):
    pass


def _matches_type(value: Any, expected: str) -> bool:
    if expected == "string":
        return isinstance(value, str)
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "boolean":
        return type(value) is bool
    if expected == "integer":
        return type(value) is int
    if expected == "number":
        if type(value) is int:
            return True
        return type(value) is float and math.isfinite(value)
    if expected == "null":
        return value is None
    return False


def _describe_type(value: Any) -> str:
    if value is None:
        return "null"
    if type(value) is bool:
        return "boolean"
    if type(value) is int:
        return "integer"
    if type(value) is float:
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def _validate(value: Any, schema: dict[str, Any], location: str) -> None:
    expected = schema.get("type")
    if expected is not None:
        allowed = [expected] if isinstance(expected, str) else list(expected)
        if not any(_matches_type(value, item) for item in allowed):
            rendered = " or ".join(allowed)
            raise ToolArgumentError(
                f"{location} must be {rendered}, got {_describe_type(value)}"
            )

    if "enum" in schema and value not in schema["enum"]:
        raise ToolArgumentError(f"{location} must be one of {schema['enum']!r}")

    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            raise ToolArgumentError(
                f"{location} must contain at least {schema['minLength']} characters"
            )
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            raise ToolArgumentError(
                f"{location} exceeds the {schema['maxLength']}-character limit"
            )

    if isinstance(value, dict):
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        missing = [name for name in required if name not in value]
        if missing:
            raise ToolArgumentError(
                f"missing required argument(s): {', '.join(sorted(missing))}"
            )
        unknown = [key for key in value if not isinstance(key, str) or key not in properties]
        # Tool calls never have a legitimate reason to include undeclared
        # parameters.  Enforce this even if an older schema omitted the
        # `additionalProperties: false` annotation.
        if unknown:
            rendered = ", ".join(sorted(repr(key) for key in unknown))
            raise ToolArgumentError(f"unknown argument(s): {rendered}")
        for key, item in value.items():
            _validate(item, properties[key], f"argument '{key}'")

    if isinstance(value, list) and "items" in schema:
        for index, item in enumerate(value):
            _validate(item, schema["items"], f"{location}[{index}]")


def validate_tool_arguments(tool_schema: dict[str, Any], args: Any) -> dict[str, Any]:
    """Validate and return args, raising a concise error on any mismatch."""

    parameters = tool_schema.get("function", {}).get("parameters", {"type": "object"})
    _validate(args, parameters, "arguments")
    return args
