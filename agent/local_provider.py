"""Capability negotiation and structured diagnostics for local providers."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import hashlib
import re
from typing import Any, Mapping, Sequence
import json
import urllib.error
import urllib.request

from .models import utc_now


@dataclass(frozen=True, slots=True)
class NormalizationReceiptV1:
    """Audit record for semantic-preserving model transport normalization."""

    tool: str
    input_fingerprint: str
    output_fingerprint: str
    actions: tuple[str, ...] = ()
    version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "input_fingerprint": self.input_fingerprint,
            "output_fingerprint": self.output_fingerprint,
            "actions": list(self.actions),
            "version": self.version,
        }


@dataclass(frozen=True, slots=True)
class ActionProposalNormalizationV1:
    """Auditable, syntax-only normalization of one textual action proposal."""

    input_fingerprint: str
    output_fingerprint: str = ""
    action_name: str = ""
    actions: tuple[str, ...] = ()
    json_error: str = ""
    delimiter_mismatch: str = ""
    version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_fingerprint": self.input_fingerprint,
            "output_fingerprint": self.output_fingerprint,
            "action_name": self.action_name,
            "actions": list(self.actions),
            "json_error": self.json_error,
            "delimiter_mismatch": self.delimiter_mismatch,
            "version": self.version,
        }


def _payload_fingerprint(value: Mapping[str, Any]) -> str:
    import hashlib

    encoded = json.dumps(
        dict(value), ensure_ascii=False, sort_keys=True, default=str,
        separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def extract_first_json_object(text: str) -> Mapping[str, Any] | None:
    """Extract one balanced JSON object without trusting surrounding prose."""
    source = str(text or "")
    decoder = json.JSONDecoder()
    fallback: Mapping[str, Any] | None = None
    for index, character in enumerate(source):
        if character != "{":
            continue
        try:
            value, _end = decoder.raw_decode(source[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, Mapping):
            # Prefer an action envelope when the provider emitted more than
            # one JSON object.  Weak local models occasionally leave the
            # outer ``}`` off an otherwise valid ``{name, arguments}``
            # response; in that case the decoder can only see the nested
            # arguments object, so keep it as a fallback while we try a
            # bounded delimiter repair below.
            # A typed ULTRA response legitimately contains nested objects with
            # display ``name`` fields (for example architecture components).
            # Only prefer a nested object when it has the complete action
            # envelope shape; otherwise keep the outer response as fallback.
            has_action_name = any(key in value for key in ("name", "tool", "action"))
            has_action_arguments = any(
                key in value for key in ("args", "arguments", "parameters")
            )
            if has_action_name and has_action_arguments:
                return value
            if fallback is None:
                fallback = value

    # A truncated model response is recoverable when the only damage is a
    # missing closing delimiter.  Do not attempt broad text rewriting: count
    # braces while respecting JSON strings, and only repair an object that
    # advertises an action envelope near its beginning.  Typed validation in
    # the caller still owns the semantic contract.
    stripped = source.lstrip()
    if stripped.startswith("{") and re.search(r'"(?:name|tool|action)"\s*:', stripped[:240]):
        depth = 0
        in_string = False
        escaped = False
        for character in stripped:
            if in_string:
                if escaped:
                    escaped = False
                elif character == "\\":
                    escaped = True
                elif character == '"':
                    in_string = False
                continue
            if character == '"':
                in_string = True
            elif character == "{":
                depth += 1
            elif character == "}" and depth:
                depth -= 1
        if depth > 0:
            try:
                value, _end = decoder.raw_decode(stripped + ("}" * depth))
            except json.JSONDecodeError:
                pass
            else:
                if isinstance(value, Mapping):
                    return value
    return fallback


def repair_structured_json_object(text: str) -> tuple[Mapping[str, Any] | None, tuple[str, ...]]:
    """Apply bounded transport repairs to a weak-model JSON envelope.

    Semantic fields are never invented here; typed validation remains the
    caller's job. The repairs cover only malformed envelope keys, trailing
    commas, and illegal control characters observed from local providers.
    """

    source = str(text or "").strip()
    if source.startswith("```"):
        source = re.sub(r"^```(?:json)?\s*", "", source, flags=re.IGNORECASE)
        source = re.sub(r"\s*```$", "", source)
    direct = extract_first_json_object(source)
    if direct is not None:
        return direct, ()
    actions: list[str] = []
    repaired = source
    key_pattern = re.compile(
        r'"(payload|summary|reasoning_summary|insights|tool_calls|artifacts|evidence|findings|issues|test_results)\\+"\s*:'
    )
    normalized = key_pattern.sub(lambda match: f'"{match.group(1)}":', repaired)
    if normalized != repaired:
        repaired = normalized
        actions.append("removed stray backslash from a known response-envelope key")
    # JSON strings use double quotes, so ``\\'`` is never a legal escape. It
    # is a frequent artifact of local models copying shell/Python snippets
    # into a verification string; remove only that invalid escape so the
    # authored apostrophe remains intact.
    normalized = repaired.replace("\\'", "'")
    if normalized != repaired:
        repaired = normalized
        actions.append("removed invalid single-quote JSON escapes")
    normalized = re.sub(r",\s*([}\]])", r"\1", repaired)
    if normalized != repaired:
        repaired = normalized
        actions.append("removed trailing JSON comma")
    normalized = "".join(char for char in repaired if char in "\r\n\t" or ord(char) >= 32)
    if normalized != repaired:
        repaired = normalized
        actions.append("removed invalid JSON control characters")
    return extract_first_json_object(repaired), tuple(actions)


def normalize_action_proposal(value: Mapping[str, Any]) -> tuple[str, dict[str, Any]] | None:
    name = str(value.get("name") or value.get("tool") or value.get("action") or "").strip()
    args = value.get("args", value.get("arguments", {}))
    if not name or not isinstance(args, Mapping):
        return None
    return name, dict(args)


def _text_fingerprint(value: str) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def _single_action_from_value(value: Any) -> tuple[Mapping[str, Any] | None, tuple[str, ...], str]:
    if isinstance(value, Mapping):
        nested = value.get("tool_calls")
        if (
            normalize_action_proposal(value) is None
            and isinstance(nested, Sequence)
            and not isinstance(nested, (str, bytes, bytearray))
        ):
            if len(nested) != 1:
                return None, (), "tool_calls must contain exactly one action"
            item = nested[0]
            if not isinstance(item, Mapping):
                return None, (), "single tool_calls item must be an object"
            name = str(
                item.get("tool_name")
                or item.get("name")
                or item.get("tool")
                or item.get("action")
                or ""
            ).strip()
            args = item.get("parameters", item.get("arguments", item.get("args", {})))
            if not name or not isinstance(args, Mapping):
                return None, (), "nested action must contain a name and object arguments"
            return {
                "name": name,
                "args": dict(args),
            }, ("unwrapped single nested tool_calls action",), ""
        return value, (), ""
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if len(value) != 1:
            return None, (), "action array must contain exactly one item"
        if not isinstance(value[0], Mapping):
            return None, (), "single action array item must be an object"
        return value[0], ("unwrapped single-item action array",), ""
    return None, (), "action transport must be an object or a single-item array"


def _repair_mismatched_single_action_array(source: str) -> tuple[str | None, tuple[str, ...], str]:
    """Repair only the observed ``[{name,args...}]]`` delimiter substitution.

    The repair is intentionally narrow: the text must begin with one action
    envelope, all nested delimiters before the final mismatch must balance, and
    the only tolerated damage is ``]`` closing an action object followed by one
    redundant trailing ``]``. No keys or semantic values are created.
    """

    stripped = str(source or "").strip()
    if not re.match(r'^\[\s*\{\s*"(?:name|tool|action)"\s*:', stripped):
        return None, (), ""
    stack: list[str] = []
    in_string = False
    escaped = False
    mismatch_index: int | None = None
    extra_index: int | None = None
    for index, character in enumerate(stripped):
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
            continue
        if character in "[{":
            stack.append(character)
            continue
        if character not in "]}":
            continue
        expected = "[" if character == "]" else "{"
        if stack and stack[-1] == expected:
            stack.pop()
            continue
        remainder = stripped[index + 1 :]
        if (
            character == "]"
            and stack == ["[", "{"]
            and re.fullmatch(r"\s*\]\s*", remainder)
        ):
            mismatch_index = index
            extra_index = index + 1 + remainder.rfind("]")
            stack.clear()
            break
        return None, (), f"unexpected {character!r} at offset {index}"
    if mismatch_index is None or extra_index is None:
        return None, (), ""
    repaired = (
        stripped[:mismatch_index]
        + "}]"
        + stripped[mismatch_index + 1 : extra_index]
        + stripped[extra_index + 1 :]
    )
    try:
        json.loads(repaired)
    except json.JSONDecodeError as exc:
        return None, (), f"bounded delimiter repair remained invalid at offset {exc.pos}: {exc.msg}"
    return repaired, (
        "closed mismatched action object before array delimiter",
        "removed redundant trailing array delimiter",
    ), "expected '}' for action object but received ']'"


def _repair_incomplete_single_action_array(
    source: str,
) -> tuple[str | None, tuple[str, ...], str]:
    """Close only missing outer delimiters on one otherwise-complete action.

    A streamed local response can finish after the action object while omitting
    the top-level array delimiter. The prefix must identify one action and all
    semantic values/nested containers must already be complete; only the outer
    action/object transport delimiters may remain open.
    """

    stripped = str(source or "").strip()
    if not re.match(r'^\[\s*\{\s*"(?:name|tool|action)"\s*:', stripped):
        return None, (), ""
    stack: list[str] = []
    in_string = False
    escaped = False
    for index, character in enumerate(stripped):
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character in "[{":
            stack.append(character)
        elif character in "]}":
            expected = "[" if character == "]" else "{"
            if not stack or stack[-1] != expected:
                return None, (), f"unexpected {character!r} at offset {index}"
            stack.pop()
    if in_string or stack not in (["["], ["[", "{"]):
        return None, (), ""
    suffix = "]" if stack == ["["] else "}]"
    repaired = stripped + suffix
    try:
        json.loads(repaired)
    except json.JSONDecodeError as exc:
        return None, (), f"bounded outer closure remained invalid at offset {exc.pos}: {exc.msg}"
    return repaired, (
        "closed incomplete single-item action array transport",
    ), f"response ended before outer delimiter {suffix!r}"


def extract_action_proposal(
    text: str,
) -> tuple[tuple[str, dict[str, Any]] | None, ActionProposalNormalizationV1]:
    """Extract exactly one allow-listable action from weak-model text.

    This function only repairs transport syntax. The caller still validates the
    action name, arguments, schema, permissions, and execution policy.
    """

    source = str(text or "").strip()
    if source.startswith("```"):
        source = re.sub(r"^```(?:json)?\s*", "", source, flags=re.IGNORECASE)
        source = re.sub(r"\s*```$", "", source)
    input_fingerprint = _text_fingerprint(source)
    actions: tuple[str, ...] = ()
    json_error = ""
    delimiter_mismatch = ""
    candidate: Mapping[str, Any] | None = None
    normalized_source = source
    decoded_complete_transport = False
    try:
        decoded, end = json.JSONDecoder().raw_decode(source)
        if source[end:].strip():
            json_error = "unexpected trailing content after action envelope"
        else:
            decoded_complete_transport = True
            candidate, actions, json_error = _single_action_from_value(decoded)
    except json.JSONDecodeError as exc:
        json_error = f"JSON decode failed at offset {exc.pos}: {exc.msg}"

    if candidate is None:
        repaired_source, repair_actions, delimiter_mismatch = (
            _repair_mismatched_single_action_array(source)
        )
        if repaired_source is not None:
            decoded = json.loads(repaired_source)
            candidate, unwrap_actions, value_error = _single_action_from_value(decoded)
            actions = (*repair_actions, *unwrap_actions)
            normalized_source = repaired_source
            json_error = value_error

    if candidate is None:
        repaired_source, repair_actions, delimiter_mismatch = (
            _repair_incomplete_single_action_array(source)
        )
        if repaired_source is not None:
            decoded = json.loads(repaired_source)
            candidate, unwrap_actions, value_error = _single_action_from_value(decoded)
            actions = (*repair_actions, *unwrap_actions)
            normalized_source = repaired_source
            json_error = value_error

    if candidate is None and not decoded_complete_transport:
        fallback = extract_first_json_object(source)
        fallback_proposal = (
            normalize_action_proposal(fallback) if fallback is not None else None
        )
        if fallback_proposal is not None:
            candidate = fallback
            actions = (*actions, "extracted action object from bounded surrounding text")

    proposal = normalize_action_proposal(candidate) if candidate is not None else None
    if proposal is None and not json_error:
        json_error = "decoded JSON did not contain one name/args action envelope"
    return proposal, ActionProposalNormalizationV1(
        input_fingerprint=input_fingerprint,
        output_fingerprint=(
            _text_fingerprint(normalized_source) if proposal is not None else ""
        ),
        action_name=(proposal[0] if proposal is not None else ""),
        actions=actions,
        json_error=json_error,
        delimiter_mismatch=delimiter_mismatch,
    )


def normalize_generated_tool_args(name: str, args: Mapping[str, Any]) -> dict[str, Any]:
    """Repair layout escapes from weak-model tool transports without touching code strings.

    A few models emit a native ``write_file`` argument whose document separators
    are the two literal characters ``\\n``.  The JSON layer has already been
    decoded at that point, so writing the value verbatim produces a one-line,
    invalid artifact.  Only source-like full documents with almost no real line
    breaks are eligible.  Escapes inside quoted source strings remain escapes.
    """

    normalized = dict(args)
    if str(name) == "stage_component_file":
        nested = normalized.get("file")
        if isinstance(nested, Mapping):
            normalized = {**dict(nested), **normalized}
            normalized.pop("file", None)
        path = str(
            normalized.get("path")
            or normalized.get("file_path")
            or normalized.get("filepath")
            or normalized.get("filename")
            or normalized.get("name")
            or ""
        ).strip()
        content = normalized.get("content")
        if not isinstance(content, str):
            content = normalized.get(
                "source",
                normalized.get("code", normalized.get("text", normalized.get("contents", ""))),
            )
        content = str(content or "")
        role = str(normalized.get("role") or normalized.get("type") or "").casefold()
        if role not in {"implementation", "preview", "test", "asset"}:
            lowered_path = path.casefold()
            lowered_content = content.lstrip().casefold()
            if lowered_path.endswith((".html", ".htm")) or lowered_content.startswith(
                ("<!doctype html", "<html")
            ):
                role = "preview"
            elif any(marker in lowered_path for marker in (".test.", ".spec.", "/test", "tests/")):
                role = "test"
            else:
                role = "implementation"
        if not path and content:
            if role == "preview":
                path = "preview/index.html"
            elif role == "test":
                path = "test/component.test.js"
            else:
                path = "src/component.js"
        return {"path": path, "content": content, "role": role}
    if str(name) == "publish_component":
        nested = normalized.get("component")
        if isinstance(nested, Mapping):
            normalized = {**dict(nested), **normalized}
            normalized.pop("component", None)
        if not isinstance(normalized.get("interface"), Mapping):
            exports = normalized.get("exports", ())
            imports = normalized.get("imports", ())
            normalized["interface"] = {
                "exports": list(exports) if isinstance(exports, (list, tuple)) else [str(exports)] if exports else [],
                "imports": list(imports) if isinstance(imports, (list, tuple)) else [str(imports)] if imports else [],
            }
        if not isinstance(normalized.get("preview"), Mapping):
            entrypoint = str(
                normalized.get("preview_entrypoint")
                or normalized.get("entrypoint")
                or "preview/index.html"
            )
            normalized["preview"] = {"entrypoint": entrypoint}
        return normalized
    field = "content" if str(name) == "write_file" else "new_str" if str(name) == "edit_file" else ""
    if not field or not isinstance(normalized.get(field), str):
        return normalized
    source = str(normalized[field])
    lowered = source.lstrip().casefold()
    source_like = lowered.startswith(("<!doctype", "<html", "<?xml")) or any(
        token in source for token in ("\\nimport ", "\\ndef ", "\\nclass ", "\\nfunction ")
    )
    if not source_like or source.count("\n") >= 2 or source.count(r"\n") < 4:
        return normalized

    output: list[str] = []
    quote: str | None = None
    escaped = False
    index = 0
    while index < len(source):
        char = source[index]
        if quote is not None:
            output.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            index += 1
            continue
        if char in {"'", '"', "`"}:
            quote = char
            output.append(char)
            index += 1
            continue
        if char == "\\" and index + 1 < len(source):
            marker = source[index + 1]
            if marker == "n":
                output.append("\n")
                index += 2
                continue
            if marker == "r":
                if index + 3 < len(source) and source[index + 2 : index + 4] == r"\n":
                    output.append("\n")
                    index += 4
                else:
                    output.append("\r")
                    index += 2
                continue
            if marker == "t":
                output.append("\t")
                index += 2
                continue
        output.append(char)
        index += 1
    normalized[field] = "".join(output)
    return normalized


def normalize_generated_tool_payload(
    name: str,
    args: Mapping[str, Any],
    *,
    context: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], NormalizationReceiptV1]:
    """Normalize provider wire variance without supplying product semantics."""

    original = dict(args)
    normalized = normalize_generated_tool_args(name, original)
    actions: list[str] = []
    if name == "submit_semantic_route":
        exact_input = str(dict(context or {}).get("exact_latest_user_input") or "")
        raw_spans = normalized.get("authority_spans")
        if exact_input and isinstance(raw_spans, Mapping):
            reference_tokens = {
                "exact_latest_user_input",
                "$exact_latest_user_input",
                "${exact_latest_user_input}",
            }
            canonical_spans: dict[str, Any] = {}
            for effect, raw_values in raw_spans.items():
                values = [raw_values] if isinstance(raw_values, str) else raw_values
                if not isinstance(values, (list, tuple)):
                    canonical_spans[str(effect)] = raw_values
                    continue
                repaired: list[Any] = []
                for span_index, span in enumerate(values):
                    if isinstance(span, str) and span.strip() in reference_tokens:
                        repaired.append(exact_input)
                        actions.append(
                            f"/authority_spans/{effect}/{span_index} resolved from "
                            "the exact_latest_user_input transport reference"
                        )
                    else:
                        repaired.append(span)
                canonical_spans[str(effect)] = repaired
            route = str(normalized.get("route") or "").strip().casefold()
            raw_effects = normalized.get("requested_effects", ())
            if isinstance(raw_effects, Mapping):
                requested_effects = {
                    str(effect).strip().casefold()
                    for effect, enabled in raw_effects.items()
                    if bool(enabled)
                }
            elif isinstance(raw_effects, str):
                requested_effects = {raw_effects.strip().casefold()}
            elif isinstance(raw_effects, (list, tuple)):
                requested_effects = {
                    str(effect).strip().casefold() for effect in raw_effects
                }
            else:
                requested_effects = set()
            # A Goal cannot mutate before a separate plan approval.  When a
            # weak model selected an internal project effect but omitted only
            # its transport citation, bind that already-authored effect to the
            # exact request as a whole.  Never do this for bounded Actions or
            # external side effects, where a missing explicit citation remains
            # a semantic contract error.
            goal_internal_effects = {"write", "run", "install", "preview"}
            if route == "goal":
                for effect in sorted(requested_effects & goal_internal_effects):
                    current = canonical_spans.get(effect)
                    if not isinstance(current, (list, tuple)) or not any(
                        str(item) for item in current
                    ):
                        canonical_spans[effect] = [exact_input]
                        actions.append(
                            f"/authority_spans/{effect} bound to the complete exact "
                            "request for an approval-gated Goal"
                        )
            # Do not carry an empty-effect sentinel into the semantic route
            # validator.  This is a wire-shape cleanup; it does not add an
            # effect or broaden authority.
            raw_effect_values = normalized.get("requested_effects")
            if isinstance(raw_effect_values, list):
                cleaned_effects = [
                    item for item in raw_effect_values
                    if str(item or "").strip().casefold()
                    not in {"", "none", "no_effect", "no effects", "no requested effects"}
                ]
                if cleaned_effects != raw_effect_values:
                    actions.append("/requested_effects empty sentinel removed")
                normalized["requested_effects"] = cleaned_effects
            normalized["authority_spans"] = canonical_spans
    elif name == "request_plan_input":
        questions = normalized.get("questions", ())
        if isinstance(questions, Mapping):
            normalized["questions"] = [dict(questions)]
            actions.append("/questions wrapped as an array")
        elif isinstance(questions, tuple):
            normalized["questions"] = list(questions)
            actions.append("/questions tuple converted to an array")
        normalized_questions: list[Any] = []
        for question_index, question in enumerate(
            normalized.get("questions", ()) or ()
        ):
            if not isinstance(question, Mapping):
                normalized_questions.append(question)
                continue
            item = dict(question)
            if "allow_freeform" not in item:
                for alias in ("allow_free_form", "allowFreeform"):
                    if alias in item:
                        item["allow_freeform"] = item.pop(alias)
                        actions.append(
                            f"/questions/{question_index}/{alias} normalized to allow_freeform"
                        )
                        break
            normalized_questions.append(item)
        if normalized_questions:
            normalized["questions"] = normalized_questions
    elif name == "propose_semantic_goal":
        raw_effects = normalized.get("requested_effects", ())
        if isinstance(raw_effects, Mapping):
            raw_effects = [
                key for key, enabled in raw_effects.items() if bool(enabled)
            ]
            actions.append("/requested_effects boolean object converted to an array")
        elif isinstance(raw_effects, str):
            raw_effects = [raw_effects]
            actions.append("/requested_effects scalar converted to an array")
        if isinstance(raw_effects, (list, tuple)):
            from .semantic import RequestedEffect

            canonical: list[Any] = []
            for effect_index, effect in enumerate(raw_effects):
                if str(effect or "").strip().casefold() in {
                    "", "none", "no_effect", "no effects", "no requested effects"
                }:
                    actions.append(
                        f"/requested_effects/{effect_index} empty sentinel removed"
                    )
                    continue
                try:
                    value = RequestedEffect.parse(effect).value
                except (TypeError, ValueError):
                    value = effect
                if value != effect:
                    actions.append(
                        f"/requested_effects/{effect_index} normalized to {value}"
                    )
                if value not in canonical:
                    canonical.append(value)
            normalized["requested_effects"] = canonical
    elif name == "submit_plan_review":
        issues = normalized.get("issues", ())
        if isinstance(issues, Mapping):
            normalized["issues"] = [dict(issues)]
            actions.append("/issues wrapped as an array")
        elif isinstance(issues, str):
            normalized["issues"] = [issues]
            actions.append("/issues scalar converted to an array")
    elif name == "propose_plan_change":
        # A plan revision describes future work; task lifecycle is owned by the
        # harness and is never model-editable.  Removing these fields is a
        # mechanical authority-boundary normalization: no title, requirement,
        # path, effect, criterion, dependency, or verification is supplied or
        # changed here.
        lifecycle_fields = {
            "status", "attempt", "attempts", "evidence", "note",
            "blocked_reason", "last_error", "started_at", "completed_at",
            "ready_at", "updated_at", "worker_id",
            # Resource leases and execution metadata are derived by the
            # harness from the accepted plan.  Weak models sometimes echo
            # these fields from the execution state when proposing a repair;
            # accepting them would both fail the transport schema and blur the
            # authority boundary.
            "resource_claims", "resource_claim", "resolved_paths", "lease",
            "execution_state", "runtime_state", "worker_state",
        }
        for field in sorted(lifecycle_fields):
            if field in normalized:
                normalized.pop(field)
                actions.append(
                    f"/{field} removed (harness-owned plan-change metadata)"
                )
        raw_tasks = normalized.get("tasks", ())
        if isinstance(raw_tasks, tuple):
            raw_tasks = list(raw_tasks)
            normalized["tasks"] = raw_tasks
            actions.append("/tasks tuple converted to an array")
        if isinstance(raw_tasks, list):
            clean_tasks: list[Any] = []
            for task_index, task in enumerate(raw_tasks):
                if not isinstance(task, Mapping):
                    clean_tasks.append(task)
                    continue
                clean = dict(task)
                if "id" not in clean and "task_id" in clean:
                    clean["id"] = clean.pop("task_id")
                    actions.append(
                        f"/tasks/{task_index}/task_id normalized to id"
                    )
                # Preserve model-authored task meaning while accepting the
                # common aliases emitted by local/tool-capable providers.  No
                # product content is invented here: each fallback is copied
                # from a field the model already supplied.
                aliases = (
                    ("name", "title"),
                    ("summary", "description"),
                    ("task", "description"),
                    ("acceptance", "acceptance_criteria"),
                    ("criteria", "acceptance_criteria"),
                    ("verification_steps", "verification"),
                    ("dependencies", "depends_on"),
                )
                for source, target in aliases:
                    if target not in clean and source in clean:
                        value = clean.pop(source)
                        if source in {"acceptance", "criteria", "verification_steps", "dependencies"} and isinstance(value, str):
                            value = [value]
                        clean[target] = value
                        actions.append(
                            f"/tasks/{task_index}/{source} normalized to {target}"
                        )
                # Some models provide only a detailed description.  A short
                # title derived from that same authored text is a structural
                # repair, not a semantic guess, and prevents a repair turn from
                # failing solely on the required display label.
                if not str(clean.get("title") or "").strip() and str(clean.get("description") or "").strip():
                    description = " ".join(str(clean["description"]).split())
                    clean["title"] = description[:180].rstrip() or "Repair task"
                    actions.append(
                        f"/tasks/{task_index}/title derived from description"
                    )
                for field in sorted(lifecycle_fields):
                    if field in clean:
                        clean.pop(field)
                        actions.append(
                            f"/tasks/{task_index}/{field} removed "
                            "(harness-owned lifecycle field)"
                        )
                clean_tasks.append(clean)
            normalized["tasks"] = clean_tasks
    elif name == "apply_patch":
        if "base_path" in normalized and not str(
            normalized.get("base_path") or ""
        ).strip():
            normalized["base_path"] = "."
            actions.append("/base_path empty optional value normalized to default '.'")
    before = _payload_fingerprint(original)
    after = _payload_fingerprint(normalized)
    if before != after and not actions:
        actions.append("provider argument layout normalized")
    return normalized, NormalizationReceiptV1(
        tool=str(name),
        input_fingerprint=before,
        output_fingerprint=after,
        actions=tuple(actions),
    )


class ProviderFailureKind(str, Enum):
    DNS_OR_SOCKET = "dns_or_socket_failure"
    CONNECTION_REFUSED = "connection_refused"
    TIMEOUT = "request_timeout"
    HTTP_4XX = "http_4xx"
    HTTP_5XX = "http_5xx"
    ENDPOINT_NOT_FOUND = "endpoint_not_found"
    MODEL_NOT_INSTALLED = "model_not_installed"
    MODEL_LOAD_FAILED = "model_load_failed"
    INVALID_PAYLOAD = "invalid_request_payload"
    UNSUPPORTED_PARAMETER = "unsupported_parameter"
    UNSUPPORTED_TOOLS = "unsupported_tool_calling"
    UNSUPPORTED_STRUCTURED_OUTPUT = "unsupported_structured_output"
    CONTEXT_LIMIT = "context_limit_exceeded"
    MALFORMED_STREAM = "malformed_streamed_response"
    INVALID_TYPED_OUTPUT = "invalid_typed_output"


_SECRET = re.compile(r"(?i)(api[_-]?key|token|password|authorization)(\s*[:=]\s*)([^\s,}\]]+)")


def redact_provider_message(value: str) -> str:
    return _SECRET.sub(lambda m: m.group(1) + m.group(2) + "[REDACTED]", str(value))


@dataclass(frozen=True, slots=True)
class ProviderDiagnostic:
    reachable: bool
    kind: ProviderFailureKind
    operation: str
    status_code: int | None = None
    provider_message: str = ""
    endpoint: str = ""
    incompatible_field: str | None = None


class ProviderRequestError(RuntimeError):
    def __init__(self, diagnostic: ProviderDiagnostic):
        self.diagnostic = diagnostic
        status = f" HTTP {diagnostic.status_code}" if diagnostic.status_code else ""
        super().__init__(f"Ollama request rejected{status}: {diagnostic.provider_message}".strip())


@dataclass(frozen=True, slots=True)
class ModelCapabilityProfile:
    model_name: str
    provider: str = "ollama"
    base_url: str = "http://localhost:11434"
    api_protocol: str = "native_chat"
    endpoint: str = "/api/chat"
    chat_support: bool = True
    completion_support: bool = False
    tool_call_support: bool = False
    structured_output_support: bool = False
    vision_support: bool = False
    embedding_support: bool = False
    streaming_support: bool = True
    thinking_support: bool = False
    context_size: int | None = None
    maximum_output_size: int | None = None
    known_unsupported_parameters: tuple[str, ...] = ()
    health_status: str = "unknown"
    last_successful_probe: datetime | None = None
    probe_evidence: Mapping[str, Any] = field(default_factory=dict)
    model_fingerprint: str = ""


class OllamaRequestCompiler:
    def compile(self, profile: ModelCapabilityProfile, *, messages: list[dict[str, Any]], tools=(), stream=True, structured=False, options: Mapping[str, Any] | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {"model": profile.model_name, "messages": messages, "stream": bool(stream)}
        unsupported = set(profile.known_unsupported_parameters)
        if tools and profile.tool_call_support and "tools" not in unsupported:
            payload["tools"] = list(tools)
        if structured and profile.structured_output_support and "format" not in unsupported:
            payload["format"] = "json"
        for key, value in (options or {}).items():
            if key not in unsupported:
                payload[key] = value
        return payload


class OllamaHandshake:
    """Probe only safe metadata and minimal generation endpoints."""

    def __init__(self, base_url: str = "http://localhost:11434", timeout: float = 10):
        self.base_url, self.timeout = base_url.rstrip("/"), timeout

    def _json(self, path: str, payload: Mapping[str, Any] | None = None) -> tuple[int, Any]:
        data = None if payload is None else json.dumps(payload).encode()
        request = urllib.request.Request(self.base_url + path, data=data, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return int(response.status), json.loads(response.read().decode())
        except urllib.error.HTTPError as error:
            body = error.read().decode(errors="replace")
            raise ProviderRequestError(ProviderDiagnostic(True, ProviderFailureKind.HTTP_4XX if error.code < 500 else ProviderFailureKind.HTTP_5XX, "probe", error.code, redact_provider_message(body), self.base_url + path)) from error
        except urllib.error.URLError as error:
            reason = error.reason
            if isinstance(reason, (TimeoutError, __import__("socket").timeout)):
                kind = ProviderFailureKind.TIMEOUT
            elif isinstance(reason, ConnectionRefusedError) or "refused" in str(reason).casefold():
                kind = ProviderFailureKind.CONNECTION_REFUSED
            else:
                kind = ProviderFailureKind.DNS_OR_SOCKET
            raise ProviderRequestError(ProviderDiagnostic(False, kind, "probe", provider_message=redact_provider_message(str(reason)), endpoint=self.base_url + path)) from error

    def probe(self, model: str) -> ModelCapabilityProfile:
        version_status, version = self._json("/api/version")
        tags_status, tags = self._json("/api/tags")
        models = {str(item.get("name") or item.get("model")): item for item in tags.get("models", [])}
        if model not in models:
            raise ProviderRequestError(ProviderDiagnostic(True, ProviderFailureKind.MODEL_NOT_INSTALLED, "model_lookup", 404, f"model {model!r} is not installed", self.base_url + "/api/tags"))
        metadata = models[model]
        capabilities = set(metadata.get("capabilities") or ())
        details = metadata.get("details") or {}
        return ModelCapabilityProfile(
            model_name=model, base_url=self.base_url, endpoint="/api/chat", api_protocol="native_chat", chat_support="completion" in capabilities,
            completion_support="completion" in capabilities, tool_call_support="tools" in capabilities,
            thinking_support="thinking" in capabilities,
            vision_support="vision" in capabilities, embedding_support="embedding" in capabilities,
            context_size=details.get("context_length"), health_status="reachable",
            last_successful_probe=utc_now(), model_fingerprint=str(metadata.get("digest") or ""),
            probe_evidence={"base_url": self.base_url, "version_status": version_status,
                            "tags_status": tags_status, "ollama_version": version.get("version"),
                            "capabilities": sorted(capabilities)},
        )
