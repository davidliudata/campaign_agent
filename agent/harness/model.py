"""Model clients: the loop talks to a protocol, never to an SDK.

`ModelTurn` is the normalized result of one model call: content blocks as plain
dicts (the exact shape the Messages API accepts back as assistant content), the
stop reason, and token usage. Two implementations:

    AnthropicClient   the real thing: Messages API tool use, with the system prompt
                      and tool definitions marked for prompt caching (they are
                      identical on every turn of a run, so turns 2..n read them
                      from cache)
    ScriptedClient    a deterministic "model" driven by a Python policy over the
                      transcript. Used by every offline test, and by the
                      rule-based investigator baseline that the LLM has to beat.

Keeping the SDK behind this seam is what lets the whole agent stack, harness and
evals included, run in CI with no API key.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol


@dataclass
class ModelTurn:
    content: list[dict]
    stop_reason: str = "end_turn"
    input_tokens: int = 0
    output_tokens: int = 0

    def tool_uses(self) -> list[dict]:
        return [b for b in self.content if b.get("type") == "tool_use"]

    def text(self) -> str:
        return "\n".join(b["text"] for b in self.content if b.get("type") == "text")


class ModelClient(Protocol):
    def complete(self, *, system: str, tools: list[dict], messages: list[dict],
                 max_tokens: int) -> ModelTurn: ...


# ── the real model ────────────────────────────────────────────────────────────

def _block_to_dict(b: Any) -> dict:
    if b.type == "text":
        return {"type": "text", "text": b.text}
    if b.type == "tool_use":
        return {"type": "tool_use", "id": b.id, "name": b.name, "input": dict(b.input)}
    # anything else (e.g. thinking blocks) must be passed back verbatim
    return b.model_dump(exclude_none=True) if hasattr(b, "model_dump") else dict(b)


@dataclass
class AnthropicClient:
    model: str
    client: Any = None
    cache: bool = True

    def __post_init__(self):
        if self.client is None:
            import anthropic                      # lazy: offline runs never import it
            self.client = anthropic.Anthropic(max_retries=3)

    def complete(self, *, system, tools, messages, max_tokens) -> ModelTurn:
        sys_blocks = [{"type": "text", "text": system}]
        tools = [dict(t) for t in tools]
        if self.cache:
            sys_blocks[0]["cache_control"] = {"type": "ephemeral"}
            if tools:
                tools[-1]["cache_control"] = {"type": "ephemeral"}
        resp = self.client.messages.create(
            model=self.model, max_tokens=max_tokens, system=sys_blocks,
            tools=tools, messages=messages,
        )
        u = resp.usage
        tokens_in = (u.input_tokens + (getattr(u, "cache_read_input_tokens", 0) or 0)
                     + (getattr(u, "cache_creation_input_tokens", 0) or 0))
        return ModelTurn([_block_to_dict(b) for b in resp.content], resp.stop_reason,
                         tokens_in, u.output_tokens)


# ── a deterministic stand-in ──────────────────────────────────────────────────

_ids = itertools.count(1)


def tool_call(name: str, input: dict, text: str = "", tokens: tuple[int, int] = (0, 0)) -> ModelTurn:
    """A turn that calls one tool (optionally with some reasoning text first)."""
    blocks = [{"type": "text", "text": text}] if text else []
    blocks.append({"type": "tool_use", "id": f"toolu_s{next(_ids):05d}", "name": name, "input": input})
    return ModelTurn(blocks, "tool_use", *tokens)


def final_text(text: str) -> ModelTurn:
    return ModelTurn([{"type": "text", "text": text}], "end_turn")


@dataclass
class ScriptedClient:
    """`policy(messages) -> ModelTurn`. Sees exactly what a model would see."""
    policy: Callable[[list[dict]], ModelTurn]
    calls: list[int] = field(default_factory=list)

    def complete(self, *, system, tools, messages, max_tokens) -> ModelTurn:
        self.calls.append(len(messages))
        return self.policy(messages)


def tool_results(messages: list[dict]) -> dict[str, dict]:
    """tool_use_id -> {name, input, content, is_error}, reconstructed from a transcript
    (content without the harness's id prefix).
    Handy for scripted policies, which must decide from the transcript alone."""
    uses: dict[str, dict] = {}
    for m in messages:
        if m["role"] == "assistant":
            for b in m["content"]:
                if b.get("type") == "tool_use":
                    uses[b["id"]] = {"name": b["name"], "input": b["input"]}
        elif isinstance(m["content"], list):
            for b in m["content"]:
                if b.get("type") == "tool_result" and b["tool_use_id"] in uses:
                    content = b["content"]
                    if isinstance(content, str) and content.startswith("[tool_use_id="):
                        content = content.split("\n", 1)[1] if "\n" in content else ""
                    uses[b["tool_use_id"]].update(content=content,
                                                   is_error=b.get("is_error", False))
    return uses
