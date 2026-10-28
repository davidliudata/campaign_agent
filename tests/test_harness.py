"""Specification for the agent harness.

The harness turns "a model plus some functions" into something you can run
against money-adjacent systems: every model-caused failure becomes an error the
model sees and can recover from, every resource it can burn is bounded by a stop
reason, policy runs in hooks the model cannot talk its way past, and everything is
traced. All offline: a ScriptedClient stands in for the model.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from agent.harness import (
    READ, WRITE, AnthropicClient, Budget, HookResult, ModelTurn, ScriptedClient, Tool,
    ToolError, ToolRegistry, allow_tiers, final_text, run_agent, tool_call, tool_results, validate,
)

ANSWER_SCHEMA = {"type": "object", "required": ["answer"],
                 "properties": {"answer": {"type": "string", "enum": ["yes", "no"]}}}


def registry(**extra) -> ToolRegistry:
    def lookup(a):
        if a["key"] == "missing":
            raise ToolError("no such key")
        if a["key"] == "boom":
            raise RuntimeError("database on fire")
        return {"key": a["key"], "value": len(a["key"])}

    tools = [
        Tool("lookup", "look a key up", {"type": "object", "required": ["key"],
             "properties": {"key": {"type": "string"}}}, lookup),
        Tool("big", "returns a lot", {"type": "object"}, lambda a: "x" * 10_000),
        Tool("answer", "final answer", ANSWER_SCHEMA, lambda a: "recorded", terminal=True),
        Tool("delete_all", "destructive", {"type": "object"}, lambda a: "deleted", tier=WRITE),
    ]
    return ToolRegistry(tools)


def script(*turns):
    """A policy that plays the given turns in order (callables get the transcript)."""
    it = iter(turns)
    def policy(messages):
        t = next(it)
        return t(messages) if callable(t) else t
    return ScriptedClient(policy)


ALL = ["lookup", "big", "answer"]


def run(client, allowed=ALL, **kw):
    return run_agent(client, system="sys", task="task", registry=registry(), allowed=allowed, **kw)


def last_result(res) -> dict:
    return res.messages[-1]["content"][0]


# ── the happy path ────────────────────────────────────────────────────────────

def test_tool_then_terminal_answer():
    res = run(script(tool_call("lookup", {"key": "abc"}), tool_call("answer", {"answer": "yes"})))
    assert res.stop == "terminal" and res.answered
    assert res.final == {"answer": "yes"}
    assert [e.name for e in res.trace.tool_calls()] == ["lookup", "answer"]
    assert json.loads(res.trace.tool_calls()[0].output)["value"] == 3
    assert res.trace.call(res.final_tool_use_id).name == "answer"


def test_end_turn_without_a_terminal_call_is_not_an_answer():
    res = run(script(final_text("I think yes.")))
    assert res.stop == "end_turn" and res.final is None and not res.answered


# ── model mistakes become error results, and the run continues ──────────────

@pytest.mark.parametrize("bad_call, expect", [
    (tool_call("nonexistent", {}), "not available"),
    (tool_call("delete_all", {}), "not available"),               # exists, but not allowlisted
    (tool_call("lookup", {}), "missing required field 'key'"),
    (tool_call("lookup", {"key": 7}), "expected string"),
    (tool_call("answer", {"answer": "maybe"}), "is not one of"),
    (tool_call("lookup", {"key": "missing"}), "no such key"),         # ToolError
    (tool_call("lookup", {"key": "boom"}), "tool failed: RuntimeError"),  # unexpected exception
])
def test_bad_calls_are_returned_as_errors_and_the_model_recovers(bad_call, expect):
    seen = {}
    def recover(messages):
        seen["result"] = messages[-1]["content"][0]
        return tool_call("answer", {"answer": "no"})
    res = run(script(bad_call, recover))
    assert seen["result"]["is_error"] is True and expect in seen["result"]["content"]
    assert res.stop == "terminal" and res.final == {"answer": "no"}


def test_an_invalid_terminal_call_does_not_end_the_run():
    res = run(script(tool_call("answer", {"answer": "maybe"}), tool_call("answer", {"answer": "yes"})))
    assert res.stop == "terminal" and res.trace.turns == 2


def test_the_model_never_sees_tools_outside_the_allowlist():
    seen = {}
    class Spy:
        def complete(self, *, system, tools, messages, max_tokens):
            seen["tools"] = [t["name"] for t in tools]
            return final_text("done")
    run(Spy(), allowed=["lookup"])
    assert seen["tools"] == ["lookup"]


# ── budgets become stop reasons ─────────────────────────────────────────────

def test_max_steps():
    forever = ScriptedClient(lambda m: tool_call("lookup", {"key": "a"}))
    res = run(forever, budget=Budget(max_steps=4))
    assert res.stop == "max_steps" and res.trace.turns == 4 and res.final is None


def test_token_budget():
    pricey = ScriptedClient(lambda m: tool_call("lookup", {"key": "a"}, tokens=(30_000, 500)))
    res = run(pricey, budget=Budget(max_tokens=100_000))
    assert res.stop == "max_tokens" and res.trace.turns == 4


def test_wall_clock_budget():
    t = iter([0.0, 1.0, 999.0])
    res = run(ScriptedClient(lambda m: tool_call("lookup", {"key": "a"})),
              budget=Budget(max_seconds=60), clock=lambda: next(t))
    assert res.stop == "timeout" and res.trace.turns == 1


def test_model_error_is_a_stop_reason_not_a_crash():
    def broken(messages):
        raise ConnectionError("overloaded")
    res = run(ScriptedClient(broken))
    assert res.stop == "model_error" and "overloaded" in res.error


def test_tool_output_is_truncated_to_bound_context():
    res = run(script(tool_call("big", {}), final_text("ok")), budget=Budget(max_output_chars=500))
    out = res.trace.tool_calls()[0].output
    assert len(out) < 600 and "truncated" in out


# ── hooks: policy the model cannot talk its way past ────────────────────────

def test_pre_hook_denial_is_returned_and_the_call_never_runs():
    ran = []
    reg = registry()
    reg._tools["delete_all"] = Tool("delete_all", "destructive", {"type": "object"},
                                    lambda a: ran.append(1) or "deleted", tier=WRITE)
    res = run_agent(script(tool_call("delete_all", {}), tool_call("answer", {"answer": "no"})),
                    system="s", task="t", registry=reg, allowed=ALL + ["delete_all"],
                    pre_hooks=[allow_tiers(READ)])
    assert ran == []
    denied = res.trace.tool_calls()[0]
    assert denied.is_error and "denied by policy" in denied.output
    assert any(e.kind == "hook" and e.name == "deny:delete_all" for e in res.trace.events)
    assert res.stop == "terminal"


def test_post_hooks_see_every_result():
    log = []
    run(script(tool_call("lookup", {"key": "ab"}), tool_call("lookup", {"key": "missing"}),
               final_text("x")), post_hooks=[lambda c, out, err: log.append((c.name, err))])
    assert log == [("lookup", False), ("lookup", True)]


def test_first_denying_hook_wins_and_none_means_no_opinion():
    res = run(script(tool_call("lookup", {"key": "a"}), final_text("x")),
              pre_hooks=[lambda c: None, lambda c: HookResult(False, "nope"), lambda c: HookResult(True)])
    assert "nope" in res.trace.tool_calls()[0].output


# ── plumbing ────────────────────────────────────────────────────────────────

def test_tool_results_reconstructs_the_transcript():
    res = run(script(tool_call("lookup", {"key": "ab"}), tool_call("answer", {"answer": "yes"})))
    got = tool_results(res.messages)
    assert [v["name"] for v in got.values()] == ["lookup", "answer"]
    assert json.loads(next(iter(got.values()))["content"])["value"] == 2


def test_trace_exports_jsonl(tmp_path):
    res = run(script(tool_call("lookup", {"key": "ab"}), tool_call("answer", {"answer": "yes"})))
    p = tmp_path / "trace.jsonl"
    res.trace.to_jsonl(p)
    rows = [json.loads(l) for l in p.read_text().splitlines()]
    assert [r["kind"] for r in rows] == ["model", "tool", "model", "tool", "stop"]
    assert rows[-1]["name"] == "terminal"


def test_registry_rejects_duplicates_and_unknown_allowlist_entries():
    with pytest.raises(ValueError):
        ToolRegistry([Tool("a", "", {}, print), Tool("a", "", {}, print)])
    with pytest.raises(ValueError):
        registry().specs(["lookup", "ghost"])


def test_validate_covers_what_models_get_wrong():
    schema = {"type": "object", "required": ["n", "xs"], "additionalProperties": False,
              "properties": {"n": {"type": "number", "minimum": 0, "maximum": 1},
                             "xs": {"type": "array", "minItems": 1, "items": {"type": "string"}}}}
    assert validate(schema, {"n": 0.5, "xs": ["a"]}) == []
    errs = validate(schema, {"n": 2, "xs": [], "extra": 1})
    assert any("maximum" in e for e in errs) and any("at least 1" in e for e in errs)
    assert any("unexpected field" in e for e in errs)
    assert validate({"type": "number"}, True)      # booleans are not numbers


# ── the real client, against a fake SDK ─────────────────────────────────────

def test_anthropic_client_normalizes_blocks_and_marks_cache():
    sent = {}
    def create(**kw):
        sent.update(kw)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text="checking"),
                     SimpleNamespace(type="tool_use", id="toolu_1", name="lookup", input={"key": "a"})],
            stop_reason="tool_use",
            usage=SimpleNamespace(input_tokens=100, output_tokens=20,
                                  cache_read_input_tokens=900, cache_creation_input_tokens=0))
    fake = SimpleNamespace(messages=SimpleNamespace(create=create))
    c = AnthropicClient(model="claude-sonnet-5-5", client=fake)
    turn = c.complete(system="sys", tools=registry().specs(["lookup", "answer"]),
                      messages=[{"role": "user", "content": "hi"}], max_tokens=512)
    assert isinstance(turn, ModelTurn)
    assert turn.tool_uses() == [{"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {"key": "a"}}]
    assert turn.text() == "checking" and turn.input_tokens == 1000 and turn.output_tokens == 20
    assert sent["model"] == "claude-sonnet-5-5"
    assert sent["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert sent["tools"][-1]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in registry().specs(["answer"])[0]     # registry specs untouched


def test_every_result_carries_its_own_tool_use_id():
    """Regression from the first live run: the model reasoned correctly but could not
    see the ids it had to cite. Each result now starts with its id."""
    res = run(script(tool_call("lookup", {"key": "ab"}), final_text("x")))
    block = res.messages[2]["content"][0]
    assert block["content"].startswith(f"[tool_use_id={block['tool_use_id']}]\n")
    assert res.trace.tool_calls()[0].output.startswith("{")      # the trace keeps the raw output
