"""Deterministic progression for bounded run/browser/output actions.

Weak models are allowed to choose the concrete project command and browser
interactions, but they must not be responsible for remembering the lifecycle
ordering.  This coordinator derives the next phase only from tool receipts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import PurePosixPath
import re
from typing import Any, Iterable, Mapping
from urllib.parse import urlsplit


class ActionWorkflowPhase(str, Enum):
    DISCOVER = "discover"
    INSPECT_STARTUP = "inspect_startup"
    START_RUNTIME = "start_runtime"
    OPEN_BROWSER = "open_browser"
    INSPECT_BROWSER = "inspect_browser"
    CAPTURE = "capture"
    VISUAL_REVIEW = "visual_review"
    PUBLISH_OUTPUT = "publish_output"
    COMPLETE = "complete"


_STARTUP_NAMES = {
    "readme", "readme.md", "readme.txt", "package.json", "pyproject.toml",
    "requirements.txt", "pipfile", "poetry.lock", "uv.lock", "cargo.toml",
    "go.mod", "gemfile", "composer.json", "pom.xml", "build.gradle",
    "dockerfile", "docker-compose.yml", "compose.yml", "app.py", "main.py",
    "manage.py", "server.py", "vite.config.js", "vite.config.ts",
    "next.config.js", "next.config.mjs", "angular.json",
}


def _payload(output: str) -> dict[str, Any]:
    try:
        value = json.loads(str(output or ""))
    except (TypeError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def _effect_names(values: Iterable[Any]) -> frozenset[str]:
    return frozenset(
        str(getattr(value, "value", value)).strip().casefold()
        for value in values
        if str(getattr(value, "value", value)).strip()
    )


@dataclass(slots=True)
class ActionExecutionCoordinatorV1:
    """Evidence-driven state machine layered over the ordinary Action loop."""

    requested_effects: frozenset[str]
    screenshot_count: int = 0
    require_browser: bool = False
    require_visual_review: bool = False
    require_output: bool = False
    listed_files: tuple[str, ...] = ()
    read_files: set[str] = field(default_factory=set)
    package_manifests: dict[str, dict[str, Any]] = field(default_factory=dict)
    dependency_directories_checked: set[str] = field(default_factory=set)
    runtime_ready: bool = False
    runtime_url: str = ""
    process_id: str = ""
    process_ids: list[str] = field(default_factory=list)
    browser_opened: bool = False
    browser_inspected: bool = False
    browser_session_id: str = ""
    browser_url: str = ""
    visited_browser_urls: set[str] = field(default_factory=set)
    interaction_targets: list[dict[str, str]] = field(default_factory=list)
    captured_paths: list[str] = field(default_factory=list)
    captured_hashes: set[str] = field(default_factory=set)
    captured_perceptual_hashes: set[str] = field(default_factory=set)
    browser_action_pending_capture: bool = False
    visual_reviewed: bool = False
    selected_paths: list[str] = field(default_factory=list)
    output_ready: bool = False
    last_failure: str = ""
    last_failure_tool: str = ""
    runtime_blockers: tuple[str, ...] = ()
    browser_reload_required: bool = False
    page_defects: tuple[str, ...] = ()
    companion_start_attempted: bool = False
    companion_start_failure: str = ""
    degraded_runtime_warnings: tuple[str, ...] = ()

    @classmethod
    def build(
        cls,
        requested_effects: Iterable[Any],
        *,
        screenshot_count: int = 0,
        require_browser: bool = False,
        require_visual_review: bool = False,
        require_output: bool = False,
    ) -> "ActionExecutionCoordinatorV1":
        return cls(
            requested_effects=_effect_names(requested_effects),
            screenshot_count=max(0, int(screenshot_count)),
            require_browser=bool(require_browser),
            require_visual_review=bool(require_visual_review),
            require_output=bool(require_output),
        )

    @property
    def active(self) -> bool:
        return bool(
            self.requested_effects.intersection({"run", "install", "preview"})
            or self.require_browser
            or self.screenshot_count
        )

    @property
    def startup_candidates(self) -> tuple[str, ...]:
        ranked: list[tuple[int, str]] = []
        for path in self.listed_files:
            name = PurePosixPath(path).name.casefold()
            if name not in _STARTUP_NAMES and not name.startswith("readme"):
                continue
            priority = (
                0 if name.startswith("readme")
                else 1 if name in {"package.json", "pyproject.toml", "requirements.txt", "pipfile"}
                else 2
            )
            ranked.append((priority, path))
        ranked.sort(key=lambda item: (item[0], item[1].count("/"), item[1].casefold()))
        return tuple(path for _, path in ranked[:6])

    @property
    def static_html_candidates(self) -> tuple[str, ...]:
        values = [path for path in self.listed_files if path.casefold().endswith((".html", ".htm"))]
        values.sort(key=lambda path: (0 if PurePosixPath(path).name.casefold() == "index.html" else 1, path))
        return tuple(values[:4])

    @property
    def inspected_startup(self) -> bool:
        candidates = set(self.startup_candidates)
        package_candidates = {
            path for path in candidates
            if PurePosixPath(path).name.casefold() == "package.json"
        }
        if self.require_browser and package_candidates:
            return package_candidates.issubset(self.read_files)
        return bool(self.read_files.intersection(candidates))

    @property
    def unread_startup_candidates(self) -> tuple[str, ...]:
        candidates = self.startup_candidates
        package_candidates = tuple(
            path for path in candidates
            if PurePosixPath(path).name.casefold() == "package.json"
        )
        required = package_candidates if self.require_browser and package_candidates else candidates[:1]
        return tuple(path for path in required if path not in self.read_files)

    @property
    def browser_component_directories(self) -> tuple[str, ...]:
        """Rank components whose declared package graph actually serves a UI."""

        ranked: list[tuple[int, str]] = []
        frontend_packages = {
            "react", "react-dom", "vue", "@angular/core", "svelte", "next",
            "nuxt", "vite", "@vitejs/plugin-react", "webpack-dev-server",
        }
        for path, manifest in self.package_manifests.items():
            dependencies = {
                str(name).casefold()
                for section in ("dependencies", "devDependencies")
                for name in (
                    manifest.get(section, {}).keys()
                    if isinstance(manifest.get(section), Mapping) else ()
                )
            }
            scripts = manifest.get("scripts") if isinstance(manifest.get("scripts"), Mapping) else {}
            script_text = " ".join(str(value).casefold() for value in scripts.values())
            marker_count = len(dependencies.intersection(frontend_packages))
            if not marker_count and not any(
                token in script_text for token in ("vite", "next dev", "nuxt", "webpack serve", "react-scripts")
            ):
                continue
            parent = str(PurePosixPath(path).parent)
            directory = "." if parent in {"", "."} else parent
            ranked.append((-marker_count, directory))
        return tuple(item for _, item in sorted(set(ranked), key=lambda row: (row[0], row[1].casefold())))

    @property
    def manifest_directories(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                "." if str(PurePosixPath(path).parent) in {"", "."}
                else str(PurePosixPath(path).parent)
                for path in sorted(self.package_manifests)
            )
        )

    def browser_start_spec(self) -> dict[str, str]:
        """Derive the single frontend package command from inspected JSON."""

        preferred = self.browser_component_directories
        if len(preferred) != 1:
            return {}
        directory = preferred[0]
        manifest = next(
            (
                value
                for path, value in self.package_manifests.items()
                if (
                    "." if str(PurePosixPath(path).parent) in {"", "."}
                    else str(PurePosixPath(path).parent)
                ) == directory
            ),
            {},
        )
        scripts = manifest.get("scripts") if isinstance(manifest.get("scripts"), Mapping) else {}
        script = "dev" if "dev" in scripts else "start" if "start" in scripts else ""
        return (
            {"command": f"npm run {script}", "cwd": directory}
            if script else {}
        )

    def companion_start_spec(self) -> dict[str, str]:
        """Return exact companion arguments only when receipts prove them."""

        if not self.runtime_blockers or self.companion_start_attempted:
            return {}
        server_candidates = [
            item for item in self.manifest_directories
            if item not in self.browser_component_directories
        ]
        if not server_candidates:
            return {}
        directory = server_candidates[0]
        manifest = next(
            (
                value
                for path, value in self.package_manifests.items()
                if (
                    "." if str(PurePosixPath(path).parent) in {"", "."}
                    else str(PurePosixPath(path).parent)
                ) == directory
            ),
            {},
        )
        scripts = manifest.get("scripts") if isinstance(manifest.get("scripts"), Mapping) else {}
        script = "dev" if "dev" in scripts else "start" if "start" in scripts else ""
        command = f"npm run {script}" if script else ""
        blocker_text = " ".join(self.runtime_blockers)
        match = re.search(
            r"https?://(?:localhost|127\.0\.0\.1|\[::1\])(?::\d{1,5})?",
            blocker_text,
            flags=re.IGNORECASE,
        )
        readiness_url = match.group(0) if match is not None else ""
        if not command or not readiness_url:
            return {}
        return {
            "command": command,
            "cwd": directory,
            "readiness_type": "url",
            "readiness_value": readiness_url,
        }

    def companion_start_instruction(self) -> str:
        """Return an evidence-derived companion-service start instruction."""

        spec = self.companion_start_spec()
        if not spec:
            return ""
        return (
            f"Call start_process now with command={spec['command']!r}, cwd={spec['cwd']!r}, "
            f"readiness_type='url', and readiness_value={spec['readiness_value']!r}. "
            "The command, component, and failed loopback origin are already proven; do not read more source files first."
        )

    @property
    def phase(self) -> ActionWorkflowPhase:
        if not self.active:
            return ActionWorkflowPhase.COMPLETE
        needs_runtime = "run" in self.requested_effects
        if needs_runtime and not self.runtime_ready:
            if not self.listed_files:
                return ActionWorkflowPhase.DISCOVER
            if self.startup_candidates and not self.inspected_startup:
                return ActionWorkflowPhase.INSPECT_STARTUP
            return ActionWorkflowPhase.START_RUNTIME
        if self.require_browser and not self.browser_opened:
            return ActionWorkflowPhase.OPEN_BROWSER
        if self.require_browser and not self.browser_inspected:
            return ActionWorkflowPhase.INSPECT_BROWSER
        if self.screenshot_count and len(self.captured_paths) < self.screenshot_count:
            return ActionWorkflowPhase.CAPTURE
        if self.require_visual_review and not self.visual_reviewed:
            return ActionWorkflowPhase.VISUAL_REVIEW
        if self.require_output and not self.output_ready:
            return ActionWorkflowPhase.PUBLISH_OUTPUT
        return ActionWorkflowPhase.COMPLETE

    def observe(self, tool_name: str, args: Mapping[str, Any], output: str, *, ok: bool) -> None:
        name = str(tool_name)
        if not ok:
            self.last_failure = str(output or "")[:1000]
            self.last_failure_tool = name
            if name == "start_process" and self.browser_opened and self.runtime_blockers:
                directory = str(args.get("cwd") or ".").strip().replace("\\", "/")
                server_candidates = {
                    item for item in self.manifest_directories
                    if item not in self.browser_component_directories
                }
                if directory in server_candidates:
                    self.companion_start_attempted = True
                    self.companion_start_failure = self.last_failure
            if name == "browser_open" and any(
                marker in self.last_failure.casefold()
                for marker in ("err_connection_refused", "not accepting connections", "connection refused")
            ):
                # The old runtime receipt is stale. Return to runtime startup
                # instead of repeatedly launching Playwright at the dead URL.
                self.runtime_ready = False
            return
        self.last_failure = ""
        self.last_failure_tool = ""
        payload = _payload(output)
        if name == "list_files":
            files = []
            for raw in str(output or "").splitlines():
                path = raw.strip().replace("\\", "/")
                if path and not path.endswith("/") and path not in files:
                    files.append(path)
            self.listed_files = tuple(files)
        elif name == "read_file":
            path = str(args.get("path") or "").strip().replace("\\", "/")
            if path:
                self.read_files.add(path)
            if PurePosixPath(path).name.casefold() == "package.json":
                try:
                    manifest = json.loads(str(output or ""))
                except (TypeError, json.JSONDecodeError):
                    manifest = None
                if isinstance(manifest, Mapping):
                    self.package_manifests[path] = dict(manifest)
        elif name == "install_dependencies":
            status = str(payload.get("status") or "")
            if status in {"installed", "already_satisfied"}:
                directory = str(args.get("directory") or ".").strip().replace("\\", "/")
                self.dependency_directories_checked.add(directory.strip("/") or ".")
        elif name == "start_process":
            receipt_ready = bool(payload.get("ready")) and payload.get("status") == "running"
            self.process_id = str(payload.get("process_id") or self.process_id)
            if self.process_id and self.process_id not in self.process_ids:
                self.process_ids.append(self.process_id)
            readiness = payload.get("readiness") if isinstance(payload.get("readiness"), Mapping) else {}
            receipt_url = str(payload.get("readiness_url") or "")
            if not self.runtime_url or not self.browser_opened:
                self.runtime_url = receipt_url
            readiness_type = str(readiness.get("type") or args.get("readiness_type") or "")
            readiness_value = str(readiness.get("value") or args.get("readiness_value") or "")
            if not self.runtime_url and readiness_type == "url":
                self.runtime_url = readiness_value
            elif not self.runtime_url and readiness_type == "port" and readiness_value.isdigit():
                self.runtime_url = f"http://127.0.0.1:{readiness_value}"
            self.runtime_ready = receipt_ready and (bool(self.runtime_url) if self.require_browser else True)
            if receipt_ready and self.browser_opened and self.runtime_blockers:
                self.runtime_blockers = ()
                self.browser_reload_required = True
                self.browser_inspected = False
        elif name in {"run_command", "run_bash"} and not self.require_browser:
            self.runtime_ready = True
        elif name == "preview_html":
            self.runtime_ready = True
            self.browser_opened = bool(payload.get("browser_opened"))
            self.browser_inspected = bool(payload.get("verification") == "passed")
            self.runtime_url = str(payload.get("url") or self.runtime_url)
            shot = str(payload.get("screenshot_path") or "")
            if shot:
                self._observe_capture(shot, str(payload.get("sha256") or ""))
        elif name == "browser_open":
            self.browser_opened = bool(payload.get("browser_opened"))
            self.browser_session_id = str(payload.get("browser_session_id") or self.browser_session_id)
            self.runtime_url = str(payload.get("url") or self.runtime_url)
            self._observe_browser_url(payload)
            self._observe_interaction_targets(payload)
        elif name in {"browser_inspect", "browser_act"}:
            self.browser_opened = self.browser_opened or bool(payload.get("browser_opened", True))
            self.browser_session_id = str(payload.get("browser_session_id") or self.browser_session_id)
            self._observe_browser_url(payload)
            self._observe_interaction_targets(payload)
            errors = self._critical_browser_errors(payload)
            visible_text = " ".join(str(payload.get("text") or "").split())
            usable_frontend_state = bool(
                self.require_browser
                and self.screenshot_count
                and (
                    len(visible_text) >= 80
                    or len(payload.get("interaction_targets") or ()) >= 2
                )
            )
            page_defects = tuple(
                dict.fromkeys(
                    str(item).strip()
                    for item in payload.get("page_errors") or ()
                    if str(item).strip()
                )
            )
            if errors and usable_frontend_state:
                # A screenshot/output task can still deliver unaffected UI
                # states when an optional API is unavailable. Preserve the
                # exact warning for the final evidence, but do not turn a
                # healthy rendered frontend into a global startup failure.
                # Functional tasks without requested screenshots remain
                # strict and still route to the companion service.
                self.degraded_runtime_warnings = tuple(
                    dict.fromkeys((*self.degraded_runtime_warnings, *errors))
                )
                self.runtime_blockers = ()
                self.browser_reload_required = False
                self.browser_inspected = True
                if name == "browser_act":
                    self.browser_action_pending_capture = True
            elif errors:
                self.runtime_blockers = errors
                self.runtime_ready = False
                self.browser_inspected = False
            elif page_defects:
                # The managed server is still ready, but this particular UI
                # state crashed. Return to the last verified app URL and let
                # the screenshot workflow choose another working feature;
                # restarting the frontend cannot repair a React page error.
                self.page_defects = tuple(dict.fromkeys((*self.page_defects, *page_defects)))
                self.runtime_blockers = ()
                self.browser_reload_required = True
                self.browser_inspected = False
            else:
                self.runtime_blockers = ()
                self.browser_reload_required = False
                self.browser_inspected = True
                if name == "browser_act":
                    self.browser_action_pending_capture = True
        elif name == "browser_screenshot":
            self.browser_opened = self.browser_opened or bool(payload.get("browser_opened", True))
            self.browser_session_id = str(payload.get("browser_session_id") or self.browser_session_id)
            self._observe_capture(
                str(payload.get("screenshot_path") or payload.get("workspace_path") or ""),
                str(payload.get("sha256") or ""),
                str(payload.get("perceptual_hash") or ""),
            )
            self.browser_action_pending_capture = False
        elif name == "inspect_images" and payload.get("status") == "evaluated":
            evaluated = {
                str(item.get("path") or "")
                for item in payload.get("evaluations") or ()
                if isinstance(item, Mapping)
            }
            self.visual_reviewed = all(path in evaluated for path in self.captured_paths)
            self.selected_paths = list(
                dict.fromkeys(
                    str(path).strip()
                    for path in payload.get("selected") or ()
                    if str(path).strip() in evaluated
                )
            )
        elif name == "publish_output" and payload.get("status") == "ready":
            self.output_ready = True

    def _observe_interaction_targets(self, payload: Mapping[str, Any]) -> None:
        values: list[dict[str, str]] = []
        for raw in payload.get("interaction_targets") or ():
            if not isinstance(raw, Mapping):
                continue
            item = {
                str(key): str(value or "").strip()
                for key, value in raw.items()
                if str(key).strip()
            }
            if item:
                values.append(item)
        if values:
            self.interaction_targets = values

    def _observe_browser_url(self, payload: Mapping[str, Any]) -> None:
        value = str(payload.get("url") or "").strip()
        if not value:
            return
        self.browser_url = value
        parsed = urlsplit(value)
        self.visited_browser_urls.add(parsed.path or "/")

    @staticmethod
    def _perceptually_duplicate(candidate: str, existing: Iterable[str]) -> bool:
        if not candidate:
            return False
        try:
            candidate_value = int(candidate, 16)
        except ValueError:
            return False
        for value in existing:
            if len(value) != len(candidate):
                continue
            try:
                distance = (candidate_value ^ int(value, 16)).bit_count()
            except ValueError:
                continue
            if distance <= max(8, int(len(candidate) * 4 * 0.04)):
                return True
        return False

    def _observe_capture(self, path: str, digest: str, perceptual_hash: str = "") -> None:
        if not path:
            return
        fingerprint = digest.strip().casefold()
        if fingerprint and fingerprint in self.captured_hashes:
            return
        visual_fingerprint = perceptual_hash.strip().casefold()
        if self._perceptually_duplicate(
            visual_fingerprint,
            self.captured_perceptual_hashes,
        ):
            return
        if path in self.captured_paths:
            return
        self.captured_paths.append(path)
        if fingerprint:
            self.captured_hashes.add(fingerprint)
        if visual_fingerprint:
            self.captured_perceptual_hashes.add(visual_fingerprint)

    def restore_durable_screenshot(self, output: str) -> None:
        """Restore an image artifact without restoring its dead browser lease."""

        payload = _payload(output)
        self._observe_capture(
            str(payload.get("screenshot_path") or payload.get("workspace_path") or ""),
            str(payload.get("sha256") or ""),
            str(payload.get("perceptual_hash") or ""),
        )

    @staticmethod
    def _critical_browser_errors(payload: Mapping[str, Any]) -> tuple[str, ...]:
        """Return runtime-breaking errors while ignoring cosmetic 404 noise."""

        critical: list[str] = []
        for key in ("network_errors", "console_errors"):
            for raw in payload.get(key) or ():
                text = str(raw).strip()
                lowered = text.casefold()
                if not text:
                    continue
                if any(marker in lowered for marker in (
                    "err_connection_refused", "connection refused", "err_name_not_resolved",
                    "err_connection_timed_out", "http 500", "http 502", "http 503", "http 504",
                )):
                    critical.append(text)
        return tuple(dict.fromkeys(critical))

    def permitted_tools(self, base_names: Iterable[str]) -> frozenset[str]:
        base = set(base_names)
        if not self.active:
            return frozenset(base)
        phase = self.phase
        read_tools = {"list_files", "read_file", "grep"}
        lifecycle = {"poll_process", "read_process_output", "stop_process"}
        mutation_tools: set[str] = set()
        if "write" in self.requested_effects:
            mutation_tools.update({"write_file", "edit_file", "apply_patch", "materialize_artifact"})
        if phase is ActionWorkflowPhase.DISCOVER:
            # Exact user/model-proven startup commands may skip discovery.
            # Once discovery has completed, later phases narrow the surface so
            # a weak model cannot fall back to repeated list/read calls.
            wanted = read_tools | mutation_tools | {
                "install_dependencies", "run_command", "run_bash", "start_process"
            }
        elif phase is ActionWorkflowPhase.INSPECT_STARTUP:
            wanted = {"read_file", "grep"} | mutation_tools
        elif phase is ActionWorkflowPhase.START_RUNTIME:
            wanted = lifecycle | mutation_tools | {
                "install_dependencies", "start_process"
            }
            if not (self.require_browser and self.browser_start_spec()):
                wanted |= {"run_command", "run_bash"}
            # A concrete startup failure may require source/config diagnosis.
            # Before that, or when browser evidence already proves the exact
            # missing companion origin, extra repository reads are no-progress.
            if (
                self.last_failure
                and self.last_failure_tool == "start_process"
                and not self.companion_start_instruction()
            ):
                wanted |= read_tools
                # A proven companion failure may require starting a declared
                # local service (for example a database via an existing
                # compose file). Re-enable bounded commands only after the
                # failure receipt exists; semantic authority and access policy
                # still govern approval.
                wanted |= {"run_command", "run_bash"}
            if self.static_html_candidates:
                wanted.add("preview_html")
            target_directories = (
                {
                    item for item in self.manifest_directories
                    if item not in self.browser_component_directories
                }
                if self.browser_opened and self.runtime_blockers
                else set(self.browser_component_directories or self.manifest_directories)
            )
            if target_directories and target_directories.issubset(
                self.dependency_directories_checked
            ):
                wanted.discard("install_dependencies")
        elif phase is ActionWorkflowPhase.OPEN_BROWSER:
            # A generated artifact can be materialized and previewed in one
            # native model turn.  Keep the accepted write tools available for
            # that atomic hand-off; the semantic effect contract still limits
            # them to requests that explicitly authorized writes.
            wanted = lifecycle | mutation_tools | {"browser_open", "preview_html"}
        elif phase is ActionWorkflowPhase.INSPECT_BROWSER:
            wanted = lifecycle | {"browser_inspect", "browser_close"}
        elif phase is ActionWorkflowPhase.CAPTURE:
            wanted = lifecycle | {"browser_inspect", "browser_act", "browser_screenshot", "browser_close"}
        elif phase is ActionWorkflowPhase.VISUAL_REVIEW:
            wanted = lifecycle | {"inspect_images"}
        elif phase is ActionWorkflowPhase.PUBLISH_OUTPUT:
            wanted = lifecycle | {"publish_output"}
        else:
            wanted = base
        return frozenset(base.intersection(wanted))

    def rewrite_call(
        self,
        tool_name: str,
        args: Mapping[str, Any],
    ) -> tuple[dict[str, Any], str]:
        """Repair one unambiguous weak-model argument from inspected evidence.

        The coordinator may only rewrite a value when the selected component
        is already proven by manifests read in this turn.  It never guesses a
        command, installs every monorepo component, or changes a user path.
        """

        normalized = dict(args)
        common_repairs: list[str] = []
        if str(tool_name) == "inspect_images" and self.captured_paths:
            repairs: list[str] = []
            if list(normalized.get("paths") or ()) != self.captured_paths:
                normalized["paths"] = list(self.captured_paths)
                repairs.append("bound vision input to all distinct screenshot receipts")
            if not str(normalized.get("purpose") or "").strip():
                normalized["purpose"] = (
                    "Evaluate and rank the captured application states for the requested final output."
                )
                repairs.append("added the action contract's visual-review purpose")
            if not str(normalized.get("criteria") or "").strip():
                normalized["criteria"] = (
                    "Check visible correctness, clarity, composition, feature coverage, distinctness, "
                    "legibility, and the absence of loading, error, or broken UI states."
                )
                repairs.append("added deterministic screenshot acceptance criteria")
            if repairs:
                return normalized, "; ".join(repairs)
        if str(tool_name) == "publish_output" and self.selected_paths:
            expected_assets = [
                {
                    "path": path,
                    "label": PurePosixPath(path.replace("\\", "/")).stem,
                    "kind": "image",
                }
                for path in self.selected_paths
            ]
            supplied = [
                str(item.get("path") or "").strip().replace("\\", "/")
                for item in normalized.get("assets") or ()
                if isinstance(item, Mapping)
            ]
            expected = [path.replace("\\", "/") for path in self.selected_paths]
            repairs: list[str] = []
            if supplied != expected:
                normalized["assets"] = expected_assets
                repairs.append("bound Output images to the visual evaluator's selected evidence")
            if self.degraded_runtime_warnings:
                limitation = self.limitation_note
                message = str(normalized.get("message") or "").strip()
                if limitation and limitation not in message:
                    normalized["message"] = (message + "\n\n" + limitation).strip()
                    repairs.append("preserved the verified optional-service limitation in Output")
            if repairs:
                return normalized, "; ".join(repairs)
        if str(tool_name) == "browser_open" and self.runtime_ready and self.runtime_url:
            requested = str(normalized.get("url") or "").strip().rstrip("/")
            verified = self.runtime_url.strip().rstrip("/")
            if requested != verified:
                normalized["url"] = self.runtime_url
                return (
                    normalized,
                    f"used the exact readiness URL {self.runtime_url!r} returned by the managed process",
                )
        if (
            str(tool_name) in {"browser_inspect", "browser_act", "browser_screenshot", "browser_close"}
            and not str(normalized.get("browser_session_id") or "").strip()
            and self.browser_session_id
        ):
            normalized["browser_session_id"] = self.browser_session_id
            common_repairs.append(
                f"reused the only active Playwright session {self.browser_session_id!r}"
            )
        if (
            str(tool_name) == "browser_inspect"
            and self.browser_reload_required
            and self.runtime_url
            and str(normalized.get("url") or "").strip().rstrip("/")
            != self.runtime_url.strip().rstrip("/")
        ):
            normalized["url"] = self.runtime_url
            common_repairs.append(
                f"reloaded the last verified app URL {self.runtime_url!r} after runtime/page repair"
            )
            return normalized, "; ".join(common_repairs)
        if str(tool_name) == "browser_act" and self.interaction_targets:
            repaired_actions: list[dict[str, Any]] = []
            repairs: list[str] = []
            for raw_action in normalized.get("actions") or ():
                action = dict(raw_action) if isinstance(raw_action, Mapping) else {}
                name_hint = str(action.get("name") or "").strip()
                selector_hint = str(action.get("selector") or "").strip()
                if not name_hint:
                    match = re.search(
                        r"\bname\s*=\s*['\"]([^'\"]+)['\"]",
                        selector_hint,
                        flags=re.IGNORECASE,
                    )
                    name_hint = match.group(1).strip() if match is not None else ""
                target: Mapping[str, Any] | None = None
                if name_hint:
                    folded = name_hint.casefold()
                    matches = [
                        item for item in self.interaction_targets
                        if folded in {
                            str(item.get("name") or "").strip().casefold(),
                            str(item.get("text") or "").strip().casefold(),
                        }
                    ]
                    if len(matches) == 1:
                        target = matches[0]
                if target is None and selector_hint:
                    selector_matches = [
                        item for item in self.interaction_targets
                        if str(item.get("selector") or "").strip() == selector_hint
                    ]
                    if len(selector_matches) == 1:
                        target = selector_matches[0]
                    elif str(action.get("action") or "").casefold() == "click":
                        # A weak model commonly emits ``a[href=...]`` even when
                        # the inspected DOM proves several links share that
                        # destination (for example a nav item and two CTAs).
                        # Clicking any member has the same navigation meaning,
                        # but Playwright correctly rejects the ambiguous CSS.
                        # Resolve only this same-destination case to one unique
                        # accessible role/name pair from the live inventory.
                        href_match = re.fullmatch(
                            r"a\s*\[\s*href\s*=\s*['\"]([^'\"]+)['\"]\s*\]",
                            selector_hint,
                            flags=re.IGNORECASE,
                        )
                        href = (
                            href_match.group(1).replace(r"\/", "/")
                            if href_match is not None
                            else ""
                        )
                        href_matches = [
                            item for item in self.interaction_targets
                            if href and str(item.get("href") or "").strip() == href
                        ]
                        for candidate in href_matches:
                            role = str(candidate.get("role") or "").strip()
                            accessible_name = str(
                                candidate.get("name") or candidate.get("text") or ""
                            ).strip()
                            if not role or not accessible_name:
                                continue
                            same_role_name = [
                                item for item in self.interaction_targets
                                if str(item.get("role") or "").strip().casefold() == role.casefold()
                                and str(
                                    item.get("name") or item.get("text") or ""
                                ).strip().casefold() == accessible_name.casefold()
                            ]
                            if len(same_role_name) == 1:
                                target = candidate
                                break
                if target is not None:
                    exact_selector = str(target.get("selector") or "").strip()
                    resolved_name = str(
                        target.get("name") or target.get("text") or name_hint
                    ).strip()
                    if exact_selector:
                        action["selector"] = exact_selector
                        action.pop("role", None)
                        action.pop("name", None)
                        action.pop("exact", None)
                    else:
                        role = str(target.get("role") or action.get("role") or "").strip()
                        if role and resolved_name:
                            action.pop("selector", None)
                            action["role"] = role
                            action["name"] = resolved_name
                            action["exact"] = True
                    repairs.append(
                        f"resolved {resolved_name or selector_hint!r} from the current DOM inventory"
                    )
                elif (
                    str(action.get("action") or "").casefold() == "click"
                    and self.screenshot_count
                ):
                    fallback = self._next_visible_route_target()
                    if fallback is not None:
                        role = str(fallback.get("role") or "link").strip()
                        resolved_name = str(
                            fallback.get("name") or fallback.get("text") or ""
                        ).strip()
                        action.pop("selector", None)
                        action["role"] = role
                        action["name"] = resolved_name
                        action["exact"] = True
                        repairs.append(
                            f"the requested target {selector_hint or name_hint!r} is absent; "
                            f"used the next visible route {resolved_name!r} for a distinct screenshot state"
                        )
                    else:
                        # There is no safe visible route to explore. A bounded
                        # Escape is a harmless action that lets the screenshot
                        # contract capture the current verified state instead
                        # of looping on an invented selector.
                        action = {"action": "press", "key": "Escape"}
                        repairs.append(
                            f"the requested target {selector_hint or name_hint!r} is absent; "
                            "preserved the current page for screenshot evidence"
                        )
                repaired_actions.append(action)
            if repairs:
                normalized["actions"] = repaired_actions
                return normalized, "; ".join((*common_repairs, *repairs))
        if common_repairs:
            return normalized, "; ".join(common_repairs)
        if str(tool_name) == "start_process":
            companion = self.companion_start_spec()
            if self.browser_opened and companion:
                normalized.update(companion)
                return (
                    normalized,
                    "used the exact companion command and failed loopback origin proven by browser evidence",
                )
            frontend = self.browser_start_spec()
            if not self.browser_opened and frontend:
                repairs: list[str] = []
                for key, value in frontend.items():
                    if str(normalized.get(key) or "").strip() != value:
                        normalized[key] = value
                        repairs.append(f"set {key}={value!r} from the inspected frontend manifest")
                readiness_type = str(normalized.get("readiness_type") or "").casefold()
                readiness_value = str(normalized.get("readiness_value") or "").strip()
                if readiness_type == "port" and not readiness_value.isdigit():
                    match = re.search(r":(\d{1,5})(?:/|$)", readiness_value)
                    if match is not None:
                        normalized["readiness_value"] = match.group(1)
                        repairs.append("normalized the declared port to its numeric value")
                if repairs:
                    return normalized, "; ".join(repairs)
            directory = str(normalized.get("cwd") or ".").strip().replace("\\", "/")
            preferred = self.browser_component_directories
            if not self.browser_opened and directory in {"", "."} and len(preferred) == 1:
                normalized["cwd"] = preferred[0]
                return (
                    normalized,
                    f"selected inspected browser component {preferred[0]!r} instead of the workspace root",
                )
        if str(tool_name) != "install_dependencies":
            return normalized, ""
        directory = str(normalized.get("directory") or ".").strip().replace("\\", "/")
        candidates = self.manifest_directories
        preferred = self.browser_component_directories
        if (
            directory in {"", "."}
            and "." not in candidates
            and len(preferred) == 1
            and preferred[0] in candidates
        ):
            normalized["directory"] = preferred[0]
            return (
                normalized,
                f"selected inspected browser component {preferred[0]!r} instead of manifest-free workspace root",
            )
        return normalized, ""

    def _next_visible_route_target(self) -> Mapping[str, Any] | None:
        """Pick a unique, same-origin route proven by the current DOM.

        This is only a screenshot-task recovery. Functional tasks retain the
        model's requested target and fail with a precise locator error.
        """

        current_path = urlsplit(self.browser_url).path or "/"
        candidates: list[Mapping[str, Any]] = []
        for item in self.interaction_targets:
            if str(item.get("role") or "").strip().casefold() != "link":
                continue
            name = str(item.get("name") or item.get("text") or "").strip()
            href = str(item.get("href") or "").strip()
            if not name or not href or href.startswith(("#", "javascript:", "mailto:")):
                continue
            parsed = urlsplit(href)
            if parsed.scheme or parsed.netloc:
                continue
            path = parsed.path or "/"
            if path == current_path or path in self.visited_browser_urls:
                continue
            same_role_name = [
                other
                for other in self.interaction_targets
                if str(other.get("role") or "").strip().casefold() == "link"
                and str(other.get("name") or other.get("text") or "").strip().casefold()
                == name.casefold()
            ]
            if len(same_role_name) != 1:
                continue
            candidates.append(item)
        if not candidates:
            return None
        # Public Login/Register pages are generally self-contained even when
        # an optional API is down, so prefer them as visual fallback states.
        candidates.sort(
            key=lambda item: (
                0
                if str(item.get("name") or item.get("text") or "").strip().casefold()
                in {"login", "register"}
                else 1,
                str(item.get("name") or item.get("text") or "").casefold(),
            )
        )
        return candidates[0]

    def validate_call(self, tool_name: str, args: Mapping[str, Any]) -> str:
        name = str(tool_name)
        if not self.active:
            return ""
        if (
            name == "stop_process"
            and self.require_browser
            and self.runtime_ready
            and not self.browser_opened
            and not self.runtime_blockers
            and str(args.get("process_id") or "").strip() in set(self.process_ids)
        ):
            return (
                "the managed project runtime is verified healthy and is still required for this browser task; "
                "do not stop it because Playwright launch failed—retry browser_open so the controller can use "
                "another installed browser engine"
            )
        if name == "browser_open" and "run" in self.requested_effects and not self.runtime_ready:
            return "start the project with start_process and obtain ready=true before browser_open"
        if name == "browser_open" and "run" in self.requested_effects:
            requested_url = str(args.get("url") or "").strip().rstrip("/")
            verified_url = self.runtime_url.strip().rstrip("/")
            if not verified_url:
                return "browser_open requires the verified readiness_url returned by start_process"
            if requested_url != verified_url:
                return (
                    f"browser_open URL must exactly match the verified runtime URL {self.runtime_url!r}; "
                    f"received {requested_url or '<missing>'!r}"
                )
        if name == "install_dependencies":
            directory = str(args.get("directory") or ".").strip().replace("\\", "/")
            candidates = self.manifest_directories
            if directory in {"", "."} and "." not in candidates and len(candidates) > 1:
                preferred = self.browser_component_directories
                recommendation = preferred[0] if preferred else candidates[0]
                return (
                    "the project has nested dependency manifests at "
                    + ", ".join(candidates)
                    + f"; call install_dependencies with directory={recommendation!r} for the component being run"
                )
        if name == "start_process":
            readiness_type = str(args.get("readiness_type") or "none").strip().casefold()
            readiness_value = str(args.get("readiness_value") or "").strip()
            if readiness_type == "none":
                return "start_process requires a real readiness signal: port, url, or a non-empty log marker"
            if readiness_type in {"port", "url", "log"} and not readiness_value:
                return f"start_process readiness_type={readiness_type!r} requires a non-empty readiness_value"
            if self.require_browser and readiness_type not in {"port", "url"}:
                return "browser tasks require port or url readiness so the verified URL can be passed to browser_open"
            preferred = self.browser_component_directories
            cwd = str(args.get("cwd") or ".").strip().replace("\\", "/").strip("/") or "."
            if self.require_browser and not self.browser_opened and preferred and cwd not in preferred:
                return (
                    f"start the browser UI component from cwd={preferred[0]!r}; "
                    f"cwd={cwd!r} is not the inspected frontend component"
                )
        if name in {"browser_inspect", "browser_act", "browser_screenshot"} and not self.browser_opened:
            return "open the project with browser_open before browser inspection or screenshots"
        if (
            name == "browser_inspect"
            and self.browser_inspected
            and not self.browser_reload_required
        ):
            return (
                "the current Playwright page is already inspected; capture this verified state "
                "with browser_screenshot or use browser_act to reach a distinct feature state"
            )
        if name == "browser_screenshot" and self.screenshot_count and not self.browser_inspected:
            return "inspect the live Playwright page before capturing requested feature screenshots"
        if name == "browser_act" and self.browser_action_pending_capture:
            return "capture the current post-interaction browser state before performing another browser action"
        if name == "inspect_images" and len(self.captured_paths) < self.screenshot_count:
            return f"capture {self.screenshot_count} distinct browser states before visual evaluation"
        if name == "publish_output" and self.require_visual_review and not self.visual_reviewed:
            return "inspect the current screenshot bytes with inspect_images before publishing Output"
        if name == "publish_output" and self.require_visual_review and not self.selected_paths:
            return "the visual evaluator selected no acceptable image; capture a better state before publishing Output"
        return ""

    @property
    def limitation_note(self) -> str:
        if not self.degraded_runtime_warnings:
            return ""
        return (
            "Limitation: an optional companion service was unavailable during capture; "
            "only visibly complete, unaffected frontend states were selected."
        )

    def directive(self) -> str:
        phase = self.phase
        candidates = ", ".join(self.startup_candidates[:4]) or "README and the declared project manifest"
        if phase is ActionWorkflowPhase.DISCOVER:
            next_action = (
                "Call list_files once to discover the project entry point and declared startup files. "
                "If the exact declared server command is already proven in context, start_process is also valid."
            )
        elif phase is ActionWorkflowPhase.INSPECT_STARTUP:
            unread = ", ".join(self.unread_startup_candidates) or candidates
            next_action = f"Read each remaining startup file now: {unread}. Do not call list_files again."
        elif phase is ActionWorkflowPhase.START_RUNTIME:
            preferred = self.browser_component_directories
            if self.browser_opened and self.runtime_blockers:
                exact_start = self.companion_start_instruction()
                next_action = (
                    "The live page exposed a missing runtime dependency: "
                    + " | ".join(self.runtime_blockers[:3])
                    + ". "
                    + (
                        exact_start
                        if exact_start
                        else (
                            "The proven companion service already failed readiness. Inspect its startup documentation/configuration "
                            "and the saved failure evidence; do not repeat the identical start command. "
                            f"Last failure: {self.companion_start_failure[:500]}"
                            if self.companion_start_attempted
                            else "Start the inspected companion service with port or URL readiness."
                        )
                    )
                    + " "
                    "Do not capture screenshots until the page is reloaded and inspected without connection/runtime errors."
                )
            else:
                component_hint = (
                    f" The inspected browser component is {preferred[0]!r}; use it as directory/cwd."
                    if preferred else ""
                )
                next_action = (
                    "Use the inspected project instructions to call start_process with a real port, URL, or log readiness signal. "
                    "Install only missing project-declared dependencies, and never open the browser before ready=true."
                    + component_hint
                )
        elif phase is ActionWorkflowPhase.OPEN_BROWSER:
            destination = self.runtime_url or "the loopback readiness URL from start_process"
            next_action = f"Call browser_open with visible=true and URL {destination}."
        elif phase is ActionWorkflowPhase.INSPECT_BROWSER:
            next_action = (
                f"Call browser_inspect for session {self.browser_session_id or '<browser_session_id>'}; "
                + (
                    f"navigate to {self.runtime_url} to reload after runtime repair, then verify there are no connection/runtime errors."
                    if self.browser_reload_required
                    else "use its authoritative targets before choosing interactions."
                )
            )
        elif phase is ActionWorkflowPhase.CAPTURE:
            remaining = max(0, self.screenshot_count - len(self.captured_paths))
            next_action = (
                f"Capture {remaining} more distinct feature state(s). Use browser_act when needed, then browser_screenshot "
                "with a meaningful name. Identical image bytes do not count twice."
                + (
                    " The frontend rendered with unavailable companion-service warnings; capture only visibly complete, "
                    "unaffected states and preserve that limitation in the final Output."
                    if self.degraded_runtime_warnings else ""
                )
            )
        elif phase is ActionWorkflowPhase.VISUAL_REVIEW:
            next_action = "Call inspect_images once with every captured screenshot path, then ground selection and copy only in that result."
        elif phase is ActionWorkflowPhase.PUBLISH_OUTPUT:
            next_action = "Call publish_output with the final message, copy-ready sections, and verified screenshot assets; or write the evidence-grounded final response so the harness can publish it."
        else:
            next_action = "All deterministic action phases have evidence; provide the concise final handoff."
        completed = (
            f"listed={bool(self.listed_files)}, startup_read={self.inspected_startup}, "
            f"runtime_ready={self.runtime_ready}, browser_open={self.browser_opened}, "
            f"browser_inspected={self.browser_inspected}, screenshots={len(self.captured_paths)}/{self.screenshot_count}, "
            f"vision={self.visual_reviewed}, output={self.output_ready}"
        )
        return (
            "DETERMINISTIC ACTION COORDINATOR\n"
            f"Current phase: {phase.value}. Evidence: {completed}.\n"
            f"Next required progress: {next_action}\n"
            "Do not restart a completed phase or ask the user for a path already represented by the active workspace."
        )


__all__ = ["ActionExecutionCoordinatorV1", "ActionWorkflowPhase"]
