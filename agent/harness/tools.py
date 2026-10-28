"""Tools: what the agent may call, with a schema and a permission tier.

A tool is a name, a description the model reads, a JSON schema for its input, a
handler, and a TIER. The tier is the harness's notion of blast radius:

    read   looks at data; safe to call any number of times
    write  changes the world (a bid, a budget, a campaign's state); every call is
           routed through the pre-tool hooks, which is where policy lives

A `terminal` tool ends the run: its validated input IS the agent's answer. Making
the answer a tool call (rather than parsing free text) means it is schema-checked,
and a malformed answer comes back to the model as an error it can fix.

Handlers raise `ToolError` for a model-facing failure ("no such campaign"); any
other exception is also caught by the loop and returned as an error result, so a
buggy tool degrades one step, never the run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

READ = "read"
WRITE = "write"


class ToolError(Exception):
    """A failure the model should see and may recover from."""


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    input_schema: dict
    handler: Callable[[dict], Any]
    tier: str = READ
    terminal: bool = False

    def spec(self) -> dict:
        """The model-facing definition (Messages API `tools` entry)."""
        return {"name": self.name, "description": self.description,
                "input_schema": self.input_schema}


_TYPES = {
    "string": str, "integer": int, "number": (int, float), "boolean": bool,
    "array": list, "object": dict,
}


def validate(schema: dict, value: Any, path: str = "input") -> list[str]:
    """A deliberately small JSON-schema check: type, required, properties, items,
    enum, minimum/maximum. Enough to reject what models actually get wrong
    (missing fields, wrong types, out-of-range numbers, invented enum values)."""
    errs: list[str] = []
    t = schema.get("type")
    if t:
        py = _TYPES[t]
        if isinstance(value, bool) and t in ("integer", "number"):
            return [f"{path}: expected {t}, got boolean"]
        if not isinstance(value, py):
            return [f"{path}: expected {t}, got {type(value).__name__}"]
    if "enum" in schema and value not in schema["enum"]:
        errs.append(f"{path}: {value!r} is not one of {schema['enum']}")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errs.append(f"{path}: {value} < minimum {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errs.append(f"{path}: {value} > maximum {schema['maximum']}")
    if isinstance(value, dict):
        for req in schema.get("required", []):
            if req not in value:
                errs.append(f"{path}: missing required field {req!r}")
        props = schema.get("properties", {})
        for k, v in value.items():
            if k in props:
                errs += validate(props[k], v, f"{path}.{k}")
            elif schema.get("additionalProperties") is False:
                errs.append(f"{path}: unexpected field {k!r}")
    if isinstance(value, list) and "items" in schema:
        if "minItems" in schema and len(value) < schema["minItems"]:
            errs.append(f"{path}: needs at least {schema['minItems']} item(s)")
        for i, v in enumerate(value):
            errs += validate(schema["items"], v, f"{path}[{i}]")
    return errs


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool] = ()):
        self._tools: dict[str, Tool] = {}
        for t in tools:
            self.add(t)

    def add(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool {tool.name!r}")
        if tool.tier not in (READ, WRITE):
            raise ValueError(f"tool {tool.name!r}: unknown tier {tool.tier!r}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def specs(self, allowed: Iterable[str]) -> list[dict]:
        """Definitions for the allowlisted tools only: the model never even sees
        a tool it is not permitted to call."""
        allowed = set(allowed)
        unknown = allowed - set(self._tools)
        if unknown:
            raise ValueError(f"allowlist names unknown tools: {sorted(unknown)}")
        return [t.spec() for n, t in self._tools.items() if n in allowed]
