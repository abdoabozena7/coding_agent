"""Offline provider utilities for tests and deterministic agent demos.

``ScriptedProvider`` implements the same contract as a real model adapter but
never opens a socket.  Tests can queue complete turns (or callables that inspect
the recorded request), exercise streaming callbacks, and assert that the whole
script was consumed.
"""

from __future__ import annotations

import copy
from collections import deque
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from .providers import AssistantTurn, ProviderActivityV1, ProviderCapabilities, ToolCall, Usage
from .providers.base import coerce_tool_args, native_data, render_for_summary, tool_call_id


@dataclass(frozen=True)
class ProviderRequest:
    conversation: list[Any]
    tools: list[Any]
    system: str


@dataclass
class ScriptedTurn:
    """A turn plus the exact fragments delivered to streaming callbacks."""

    turn: AssistantTurn
    text_chunks: list[str] = field(default_factory=list)
    thought_chunks: list[str] = field(default_factory=list)


ScriptItem = (
    AssistantTurn
    | ScriptedTurn
    | str
    | Mapping[str, Any]
    | BaseException
    | Callable[[ProviderRequest], Any]
)


class ScriptedProvider:
    """Deterministic, network-free provider implementing the production API."""

    capabilities = ProviderCapabilities(
        streaming=True,
        tool_calling=True,
        thinking=True,
        tool_call_ids=True,
        native_replay=True,
    )

    def __init__(
        self,
        turns: Iterable[ScriptItem] = (),
        *,
        summaries: Iterable[str | BaseException | Callable[[list[Any]], str]] = (),
        model: str = "scripted",
    ):
        self.model = model
        self._turns = deque(turns)
        self._summaries = deque(summaries)
        self.calls: list[ProviderRequest] = []
        self.summary_calls: list[list[Any]] = []

    @property
    def remaining(self) -> int:
        return len(self._turns)

    @staticmethod
    def _coerce_usage(value: Any) -> Usage | None:
        if value is None or isinstance(value, Usage):
            return copy.deepcopy(value)
        if isinstance(value, Mapping):
            return Usage(
                input_tokens=value.get("input_tokens", 0) or 0,
                cached_tokens=value.get("cached_tokens", 0) or 0,
                output_tokens=value.get("output_tokens", 0) or 0,
            )
        return None

    @staticmethod
    def _coerce_turn(value: Any) -> ScriptedTurn:
        if isinstance(value, ScriptedTurn):
            return copy.deepcopy(value)
        if isinstance(value, AssistantTurn):
            return ScriptedTurn(copy.deepcopy(value))
        if isinstance(value, str):
            return ScriptedTurn(AssistantTurn(text=value))
        if not isinstance(value, Mapping):
            raise TypeError(f"Unsupported scripted turn: {type(value).__name__}")

        raw_calls = value.get("tool_calls") or []
        calls = []
        if isinstance(raw_calls, (list, tuple)):
            for index, call in enumerate(raw_calls):
                if isinstance(call, ToolCall):
                    calls.append(copy.deepcopy(call))
                    continue
                if not isinstance(call, Mapping):
                    continue
                name = str(call.get("name") or "unknown_tool")
                native = call.get("native")
                calls.append(
                    ToolCall(
                        id=tool_call_id("scripted", call.get("id"), index, name),
                        name=name,
                        args=coerce_tool_args(call.get("args")),
                        native=(
                            copy.deepcopy(dict(native))
                            if isinstance(native, Mapping)
                            else {}
                        ),
                    )
                )
        native = value.get("native")
        turn = AssistantTurn(
            text=value.get("text", value.get("content")),
            tool_calls=calls,
            usage=ScriptedProvider._coerce_usage(value.get("usage")),
            native=(copy.deepcopy(dict(native)) if isinstance(native, Mapping) else {}),
        )
        text_chunks = value.get("text_chunks") or []
        thought_chunks = value.get("thought_chunks") or []
        return ScriptedTurn(
            turn=turn,
            text_chunks=[str(part) for part in text_chunks],
            thought_chunks=[str(part) for part in thought_chunks],
        )

    @staticmethod
    def _native_thoughts(turn: AssistantTurn) -> list[str]:
        ollama = native_data(turn.to_message(), "ollama")
        if ollama.get("thinking"):
            return [str(ollama["thinking"])]
        gemini = native_data(turn.to_message(), "gemini")
        parts = gemini.get("parts")
        if isinstance(parts, list):
            return [
                str(part.get("text"))
                for part in parts
                if isinstance(part, Mapping)
                and part.get("type") == "thought"
                and part.get("text")
            ]
        return []

    def call(
        self, conversation, tools, system, on_text=None, on_thought=None,
        on_activity=None,
    ) -> AssistantTurn:
        request = ProviderRequest(
            conversation=copy.deepcopy(list(conversation or [])),
            tools=copy.deepcopy(list(tools or [])),
            system=str(system or ""),
        )
        self.calls.append(request)
        if on_activity:
            on_activity(ProviderActivityV1(state="request_sent"))
        if not self._turns:
            raise AssertionError("ScriptedProvider has no turn left to return")

        item = self._turns.popleft()
        if isinstance(item, BaseException):
            raise item
        if callable(item):
            item = item(request)
            if isinstance(item, BaseException):
                raise item
        scripted = self._coerce_turn(item)

        thoughts = scripted.thought_chunks or self._native_thoughts(scripted.turn)
        if on_thought:
            for fragment in thoughts:
                if on_activity:
                    on_activity(ProviderActivityV1(
                        state="receiving",
                        received_bytes=len(fragment.encode("utf-8")),
                        received_chunks=1,
                    ))
                on_thought(fragment)
        text_chunks = scripted.text_chunks
        if not text_chunks and scripted.turn.text:
            text_chunks = [scripted.turn.text]
        if on_text:
            for fragment in text_chunks:
                if on_activity:
                    on_activity(ProviderActivityV1(
                        state="receiving",
                        received_bytes=len(fragment.encode("utf-8")),
                        received_chunks=1,
                    ))
                on_text(fragment)
        if on_activity:
            usage = scripted.turn.usage
            on_activity(ProviderActivityV1(
                state="completed",
                received_tokens=(usage.output_tokens if usage is not None else 0),
            ))
        return copy.deepcopy(scripted.turn)

    def summarize(self, messages) -> str:
        snapshot = copy.deepcopy(list(messages or []))
        self.summary_calls.append(snapshot)
        if not self._summaries:
            return render_for_summary(snapshot)
        item = self._summaries.popleft()
        if isinstance(item, BaseException):
            raise item
        if callable(item):
            return str(item(snapshot))
        return str(item)

    def assert_exhausted(self) -> None:
        if self._turns or self._summaries:
            raise AssertionError(
                "ScriptedProvider still has "
                f"{len(self._turns)} turn(s) and {len(self._summaries)} summary item(s)"
            )


def semantic_turn(
    route: str,
    *,
    original: str,
    response: str = "",
    effects: Iterable[str] = (),
    needs_workspace_tools: bool | None = None,
    interpretation: str = "scripted semantic decision",
    outcome_kind: str = "",
    goal_intake: Mapping[str, Any] | None = None,
    task_demand: Mapping[str, Any] | None = None,
    uncertainty: str = "clear",
    clarification_question: str = "",
) -> dict[str, Any]:
    """Build a valid submit_semantic_turn call for provider-contract tests."""

    effect_values = tuple(str(item) for item in effects)
    mutating = {"write", "run", "install", "preview", "external_side_effect"}
    spans = {
        effect: ([str(original)] if effect in mutating else [])
        for effect in effect_values
    }
    args: dict[str, Any] = {
        "route": str(route),
        "outcome_kind": str(outcome_kind),
        "interpretation": interpretation,
        "requested_effects": {
            name: name in effect_values
            for name in ("read", "write", "run", "install", "preview", "external_side_effect")
        },
        "authority_spans": {
            name: list(spans.get(name, ()))
            for name in ("read", "write", "run", "install", "preview", "external_side_effect")
        },
        "needs_workspace_tools": bool(effect_values) if needs_workspace_tools is None else bool(needs_workspace_tools),
        "direct_response": str(response),
        "uncertainty": str(uncertainty),
        "clarification_question": str(clarification_question),
        "task_demand": dict(task_demand or {
            "reasoning": 1,
            "implementation": 1,
            "context_breadth": 1,
            "coordination": 1,
            "verification": 1,
            "visual_runtime": 1,
            "component_count": 1,
            "independently_parallelizable": False,
            "rationale": ["Scripted low-demand turn"],
        }),
    }
    if goal_intake is not None:
        args["goal_intake"] = dict(goal_intake)
    return {
        "tool_calls": [{
            "id": "semantic-turn",
            "name": "submit_semantic_turn",
            "args": args,
        }]
    }


def semantic_route(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Build the V3 route-only provider call."""

    value = semantic_turn(*args, **kwargs)
    value["tool_calls"][0]["name"] = "submit_semantic_route"
    value["tool_calls"][0]["args"].pop("goal_intake", None)
    return value


def semantic_goal_intake_turn(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "tool_calls": [{
            "id": "semantic-goal-intake",
            "name": "submit_goal_intake",
            "args": dict(value),
        }]
    }


def semantic_goal_intake(
    original: str,
    *,
    recommended_mode: str = "normal",
    questions: Iterable[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    question_values = tuple(questions)
    deep = recommended_mode == "ultra"
    level = 3 if deep else 1
    return {
        "objective": str(original),
        "deliverables": ["Complete the requested outcome"],
        "constraints": ["Preserve unrelated workspace content"],
        "exclusions": [],
        "acceptance_expectations": ["The requested behavior is implemented and verified"],
        "assumptions": [],
        "risks": [],
        "component_count": 2 if deep else 1,
        "parallelism_required": deep,
        "coordination_summary": (
            "Parallel specialist coordination" if recommended_mode == "ultra"
            else "One sequential workflow"
        ),
        "uncertainty": "clear" if not question_values else "consequential choices remain",
        "complexity_reasons": ["Model-assessed test objective"],
        "task_demand": {
            "reasoning": level,
            "implementation": level,
            "context_breadth": level,
            "coordination": level,
            "verification": level,
            "visual_runtime": 1,
            "component_count": 2 if deep else 1,
            "independently_parallelizable": deep,
            "rationale": ["Model-assessed test objective"],
        },
        "questions": [dict(item) for item in question_values],
    }


__all__ = [
    "ProviderRequest", "ScriptedProvider", "ScriptedTurn", "semantic_goal_intake",
    "semantic_goal_intake_turn", "semantic_route", "semantic_turn",
]
