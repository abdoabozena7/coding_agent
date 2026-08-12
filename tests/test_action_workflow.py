from __future__ import annotations

import json

from agent.action_workflow import ActionExecutionCoordinatorV1, ActionWorkflowPhase


def coordinator() -> ActionExecutionCoordinatorV1:
    return ActionExecutionCoordinatorV1.build(
        ("read", "run", "preview"),
        screenshot_count=3,
        require_browser=True,
        require_visual_review=True,
        require_output=True,
    )


def test_browser_action_progression_is_receipt_driven() -> None:
    flow = coordinator()
    assert flow.phase is ActionWorkflowPhase.DISCOVER

    flow.observe(
        "list_files",
        {},
        "README.md\nrequirements.txt\napp.py\n",
        ok=True,
    )
    assert flow.phase is ActionWorkflowPhase.INSPECT_STARTUP
    assert "README.md" in flow.directive()

    flow.observe("read_file", {"path": "README.md"}, "streamlit run app.py", ok=True)
    assert flow.phase is ActionWorkflowPhase.START_RUNTIME
    flow.observe(
        "start_process",
        {"readiness_type": "port", "readiness_value": "8501"},
        json.dumps({"process_id": "process-1", "status": "running", "ready": True}),
        ok=True,
    )
    assert flow.phase is ActionWorkflowPhase.OPEN_BROWSER
    assert flow.runtime_url == "http://127.0.0.1:8501"

    flow.observe(
        "browser_open",
        {"url": flow.runtime_url},
        json.dumps({
            "browser_session_id": "browser-1", "status": "running",
            "browser_opened": True, "url": flow.runtime_url,
        }),
        ok=True,
    )
    assert flow.phase is ActionWorkflowPhase.INSPECT_BROWSER
    rewritten, reason = flow.rewrite_call("browser_inspect", {})
    assert rewritten == {"browser_session_id": "browser-1"}
    assert "only active Playwright session" in reason
    assert "inspect the live" in flow.validate_call(
        "browser_screenshot", {"browser_session_id": "browser-1"}
    )

    flow.observe(
        "browser_inspect",
        {"browser_session_id": "browser-1"},
        json.dumps({
            "browser_session_id": "browser-1",
            "browser_opened": True,
            "interaction_targets": [{
                "role": "link",
                "name": "Doctors",
                "text": "Doctors",
                "href": "/doctors",
                "selector": 'a[href="/doctors"]',
            }],
        }),
        ok=True,
    )
    assert flow.phase is ActionWorkflowPhase.CAPTURE
    repaired, repair_reason = flow.rewrite_call(
        "browser_act",
        {
            "browser_session_id": "browser-1",
            "actions": [{
                "action": "click",
                "selector": "[role='link'][name='Doctors']",
            }],
        },
    )
    assert repaired["actions"][0]["selector"] == 'a[href="/doctors"]'
    assert "current DOM inventory" in repair_reason


def test_ambiguous_same_destination_link_uses_unique_accessible_target() -> None:
    flow = coordinator()
    flow.browser_session_id = "browser-1"
    flow.interaction_targets = [
        {
            "role": "link", "name": "Doctors", "text": "Doctors",
            "href": "/doctors", "selector": "",
        },
        {
            "role": "link", "name": "Find a Doctor", "text": "Find a Doctor",
            "href": "/doctors", "selector": "",
        },
        {
            "role": "link", "name": "Browse Doctors", "text": "Browse Doctors",
            "href": "/doctors", "selector": "",
        },
    ]

    repaired, reason = flow.rewrite_call(
        "browser_act",
        {
            "browser_session_id": "browser-1",
            "actions": [{"action": "click", "selector": 'a[href="/doctors"]'}],
        },
    )

    assert repaired["actions"] == [{
        "action": "click", "role": "link", "name": "Doctors", "exact": True,
    }]
    assert "current DOM inventory" in reason


def test_output_is_rewritten_to_vision_selected_images_and_preserves_limitation() -> None:
    flow = coordinator()
    flow.selected_paths = ["output/browser/doctors.png"]
    flow.degraded_runtime_warnings = ("GET http://127.0.0.1:7000/api/doctors failed",)

    repaired, reason = flow.rewrite_call(
        "publish_output",
        {
            "message": "The selected project view is ready.",
            "assets": [
                {"path": "output/browser/home.png", "kind": "image"},
                {"path": "output/browser/doctors.png", "kind": "image"},
            ],
        },
    )

    assert [item["path"] for item in repaired["assets"]] == [
        "output/browser/doctors.png"
    ]
    assert flow.limitation_note in repaired["message"]
    assert "visual evaluator's selected evidence" in reason
    assert "optional-service limitation" in reason


def test_repeated_browser_inspect_is_rejected_as_no_progress() -> None:
    flow = coordinator()
    flow.browser_opened = True
    flow.browser_inspected = True

    error = flow.validate_call("browser_inspect", {"browser_session_id": "browser-1"})

    assert "already inspected" in error
    assert "browser_screenshot" in error


def test_missing_screenshot_target_recovers_to_a_proven_visible_route() -> None:
    flow = coordinator()
    flow.screenshot_count = 3
    flow.browser_session_id = "browser-1"
    flow.browser_opened = True
    flow.browser_inspected = True
    flow.browser_url = "http://localhost:3000/doctors"
    flow.visited_browser_urls.add("/doctors")
    flow.interaction_targets = [
        {"role": "link", "name": "Home", "text": "Home", "href": "/"},
        {"role": "link", "name": "Login", "text": "Login", "href": "/login"},
    ]

    repaired, reason = flow.rewrite_call(
        "browser_act",
        {
            "actions": [{"action": "click", "selector": 'a[href="/doctors/1"]'}],
        },
    )

    assert repaired["actions"] == [{
        "action": "click", "role": "link", "name": "Login", "exact": True,
    }]
    assert "next visible route" in reason


def test_duplicate_screenshot_bytes_do_not_advance_count() -> None:
    flow = coordinator()
    flow.listed_files = ("README.md", "app.py")
    flow.read_files.add("README.md")
    flow.runtime_ready = True
    flow.browser_opened = True
    flow.browser_inspected = True
    for index, digest in enumerate(("a" * 64, "a" * 64, "b" * 64), start=1):
        flow.observe(
            "browser_screenshot",
            {"browser_session_id": "browser-1"},
            json.dumps({
                "browser_session_id": "browser-1",
                "browser_opened": True,
                "screenshot_path": f"output/browser/shot-{index}.png",
                "sha256": digest,
            }),
            ok=True,
        )

    assert flow.captured_paths == [
        "output/browser/shot-1.png",
        "output/browser/shot-3.png",
    ]
    assert flow.phase is ActionWorkflowPhase.CAPTURE
    assert "1 more distinct" in flow.directive()


def test_visually_near_duplicate_screenshots_do_not_advance_count() -> None:
    flow = coordinator()
    flow.listed_files = ("README.md", "app.py")
    flow.read_files.add("README.md")
    flow.runtime_ready = True
    flow.browser_opened = True
    flow.browser_inspected = True
    receipts = (
        ("a" * 64, "0" * 64),
        ("b" * 64, ("0" * 63) + "1"),
        ("c" * 64, "f" * 64),
    )
    for index, (digest, perceptual) in enumerate(receipts, start=1):
        flow.observe(
            "browser_screenshot",
            {"browser_session_id": "browser-1"},
            json.dumps({
                "browser_session_id": "browser-1",
                "browser_opened": True,
                "screenshot_path": f"output/browser/visual-{index}.png",
                "sha256": digest,
                "perceptual_hash": perceptual,
            }),
            ok=True,
        )

    assert flow.captured_paths == [
        "output/browser/visual-1.png",
        "output/browser/visual-3.png",
    ]


def test_each_browser_action_requires_a_capture_before_the_next_action() -> None:
    flow = coordinator()
    flow.browser_opened = True
    flow.browser_inspected = True
    flow.browser_session_id = "browser-1"
    clean_page = json.dumps({
        "browser_session_id": "browser-1",
        "browser_opened": True,
        "network_errors": [],
        "console_errors": [],
        "page_errors": [],
    })
    flow.observe("browser_act", {"browser_session_id": "browser-1"}, clean_page, ok=True)

    assert "capture the current post-interaction" in flow.validate_call(
        "browser_act", {"browser_session_id": "browser-1", "actions": []}
    )

    flow.observe(
        "browser_screenshot",
        {"browser_session_id": "browser-1"},
        json.dumps({
            "browser_session_id": "browser-1",
            "browser_opened": True,
            "screenshot_path": "output/browser/state.png",
            "sha256": "d" * 64,
            "perceptual_hash": "a" * 64,
        }),
        ok=True,
    )
    assert flow.validate_call(
        "browser_act", {"browser_session_id": "browser-1", "actions": []}
    ) == ""


def test_restored_screenshot_does_not_restore_dead_browser_session() -> None:
    flow = coordinator()
    flow.restore_durable_screenshot(json.dumps({
        "browser_session_id": "browser-dead",
        "browser_opened": True,
        "screenshot_path": "output/browser/home.png",
        "sha256": "a" * 64,
    }))

    assert flow.captured_paths == ["output/browser/home.png"]
    assert flow.browser_opened is False
    assert flow.browser_session_id == ""


def test_visual_review_arguments_are_bound_to_verified_screenshots() -> None:
    flow = coordinator()
    flow.captured_paths = ["output/home.png", "output/doctors.png", "output/login.png"]

    repaired, reason = flow.rewrite_call("inspect_images", {})

    assert repaired["paths"] == flow.captured_paths
    assert "visual-review purpose" in reason
    assert "acceptance criteria" in reason
    assert repaired["purpose"]
    assert repaired["criteria"]


def test_run_effect_permits_declared_dependency_install_but_does_not_require_it() -> None:
    flow = ActionExecutionCoordinatorV1.build(("run",), require_output=True)
    flow.listed_files = ("package.json",)
    flow.read_files.add("package.json")
    permitted = flow.permitted_tools({"list_files", "read_file", "install_dependencies", "start_process"})
    assert "install_dependencies" in permitted
    assert "start_process" in permitted
    assert "list_files" not in permitted

    flow.package_manifests["package.json"] = {
        "scripts": {"dev": "vite"}, "dependencies": {"vite": "1"}
    }
    flow.observe(
        "install_dependencies",
        {"directory": "."},
        json.dumps({"status": "already_satisfied"}),
        ok=True,
    )
    permitted = flow.permitted_tools(
        {"install_dependencies", "start_process"}
    )
    assert "install_dependencies" not in permitted


def test_browser_project_selects_frontend_component_and_rejects_false_readiness() -> None:
    flow = coordinator()
    flow.observe(
        "list_files",
        {},
        "client/package.json\nserver/package.json\n",
        ok=True,
    )
    flow.observe(
        "read_file",
        {"path": "client/package.json"},
        json.dumps({"scripts": {"dev": "vite"}, "dependencies": {"react": "1"}}),
        ok=True,
    )
    flow.observe(
        "read_file",
        {"path": "server/package.json"},
        json.dumps({"scripts": {"dev": "nodemon server.js"}, "dependencies": {"express": "1"}}),
        ok=True,
    )

    assert flow.browser_component_directories == ("client",)
    rewritten, reason = flow.rewrite_call(
        "install_dependencies", {"directory": ".", "manager": "npm"}
    )
    assert rewritten["directory"] == "client"
    assert "inspected browser component" in reason
    start_args, start_reason = flow.rewrite_call(
        "start_process",
        {
            "command": "npm run dev --prefix client",
            "cwd": "client",
            "readiness_type": "port",
            "readiness_value": "localhost:3000",
        },
    )
    assert start_args["cwd"] == "client"
    assert start_args["command"] == "npm run dev"
    assert start_args["readiness_value"] == "3000"
    assert "frontend manifest" in start_reason
    assert "directory='client'" in flow.validate_call("install_dependencies", {"manager": "npm"})
    assert "cwd='client'" in flow.validate_call(
        "start_process",
        {
            "command": "npm run dev",
            "cwd": "server",
            "readiness_type": "port",
            "readiness_value": "3000",
        },
    )
    assert "port or url readiness" in flow.validate_call(
        "start_process",
        {
            "command": "npm run dev",
            "cwd": "client",
            "readiness_type": "log",
            "readiness_value": "ready",
        },
    )
    strict_runtime_tools = flow.permitted_tools(
        {"run_command", "run_bash", "start_process"}
    )
    assert strict_runtime_tools == {"start_process"}

    flow.browser_opened = True
    flow.runtime_blockers = (
        "GET http://localhost:7000/api/doctors: net::ERR_CONNECTION_REFUSED",
    )
    instruction = flow.companion_start_instruction()
    assert "command='npm run dev'" in instruction
    assert "cwd='server'" in instruction
    assert "readiness_value='http://localhost:7000'" in instruction
    companion_args, companion_reason = flow.rewrite_call(
        "start_process", {"command": "npm run dev", "cwd": "."}
    )
    assert companion_args == {
        "command": "npm run dev",
        "cwd": "server",
        "readiness_type": "url",
        "readiness_value": "http://localhost:7000",
    }
    assert "browser evidence" in companion_reason
    assert "do not read more source files" in flow.directive()

    flow.observe(
        "start_process",
        companion_args,
        "Error: managed process did not become ready: app crashed",
        ok=False,
    )
    assert flow.companion_start_attempted is True
    assert flow.companion_start_instruction() == ""
    assert "do not repeat" in flow.directive().casefold()
    permitted = flow.permitted_tools(
        {"list_files", "read_file", "grep", "start_process", "poll_process"}
    )
    assert {"list_files", "read_file", "grep"}.issubset(permitted)


def test_browser_open_must_use_exact_verified_runtime_url() -> None:
    flow = coordinator()
    flow.listed_files = ("package.json",)
    flow.read_files.add("package.json")
    flow.runtime_ready = True
    flow.runtime_url = "http://127.0.0.1:5173"
    assert "exactly match" in flow.validate_call(
        "browser_open", {"url": "http://localhost:3000"}
    )
    repaired, reason = flow.rewrite_call(
        "browser_open", {"url": "http://localhost:3000", "visible": True}
    )
    assert repaired["url"] == "http://127.0.0.1:5173"
    assert "exact readiness URL" in reason
    assert flow.validate_call("browser_open", repaired) == ""


def test_browser_launch_failure_cannot_stop_a_verified_project_runtime() -> None:
    flow = coordinator()
    flow.listed_files = ("client/package.json",)
    flow.read_files.add("client/package.json")
    flow.runtime_ready = True
    flow.runtime_url = "http://127.0.0.1:5173"
    flow.process_id = "process-ui"
    flow.process_ids.append("process-ui")
    flow.observe(
        "browser_open",
        {"url": flow.runtime_url},
        "Error: Playwright browser launch failed: Chrome closed during startup",
        ok=False,
    )

    rejection = flow.validate_call(
        "stop_process", {"process_id": "process-ui"}
    )

    assert flow.runtime_ready is True
    assert "verified healthy" in rejection
    assert "retry browser_open" in rejection


def test_browser_connection_failure_routes_to_companion_runtime_and_reload() -> None:
    flow = coordinator()
    flow.listed_files = ("client/package.json", "server/package.json")
    flow.read_files.update(flow.listed_files)
    flow.package_manifests = {
        "client/package.json": {
            "scripts": {"dev": "vite"}, "dependencies": {"react": "1"}
        },
        "server/package.json": {
            "scripts": {"dev": "node server.js"}, "dependencies": {"express": "1"}
        },
    }
    flow.runtime_ready = True
    flow.runtime_url = "http://127.0.0.1:5173"
    flow.browser_opened = True
    flow.browser_session_id = "browser-1"

    flow.observe(
        "browser_inspect",
        {"browser_session_id": "browser-1"},
        json.dumps({
            "browser_session_id": "browser-1",
            "browser_opened": True,
            "network_errors": [
                "GET http://localhost:7000/api/doctors: net::ERR_CONNECTION_REFUSED"
            ],
            "console_errors": [],
            "page_errors": [],
        }),
        ok=True,
    )
    assert flow.phase is ActionWorkflowPhase.START_RUNTIME
    assert "server" in flow.directive()
    assert "connection" in flow.validate_call(
        "browser_screenshot", {"browser_session_id": "browser-1"}
    ).casefold() or not flow.browser_inspected

    flow.observe(
        "start_process",
        {"cwd": "server", "readiness_type": "port", "readiness_value": "7000"},
        json.dumps({
            "process_id": "process-api",
            "status": "running",
            "ready": True,
            "readiness_url": "http://127.0.0.1:7000",
        }),
        ok=True,
    )
    assert flow.runtime_url == "http://127.0.0.1:5173"
    assert flow.phase is ActionWorkflowPhase.INSPECT_BROWSER
    assert "reload" in flow.directive().casefold()

    flow.observe(
        "browser_inspect",
        {"browser_session_id": "browser-1", "url": flow.runtime_url},
        json.dumps({
            "browser_session_id": "browser-1",
            "browser_opened": True,
            "network_errors": [],
            "console_errors": [],
            "page_errors": [],
        }),
        ok=True,
    )
    assert flow.phase is ActionWorkflowPhase.CAPTURE


def test_screenshot_task_can_capture_usable_frontend_when_optional_api_is_down() -> None:
    flow = coordinator()
    flow.listed_files = ("client/package.json", "server/package.json")
    flow.read_files.update(flow.listed_files)
    flow.runtime_ready = True
    flow.runtime_url = "http://127.0.0.1:3000"
    flow.browser_opened = True
    flow.browser_session_id = "browser-1"

    flow.observe(
        "browser_inspect",
        {"browser_session_id": "browser-1"},
        json.dumps({
            "browser_session_id": "browser-1",
            "browser_opened": True,
            "text": (
                "MediCare Home Doctors Login Register Your Health, Our Priority "
                "Book appointments with trusted specialists and get the care you deserve."
            ),
            "interaction_targets": [
                {"role": "link", "name": "Login", "selector": 'a[href="/login"]'},
                {"role": "link", "name": "Register", "selector": 'a[href="/register"]'},
            ],
            "network_errors": [
                "GET http://localhost:7000/api/doctors: net::ERR_CONNECTION_REFUSED"
            ],
            "console_errors": [],
            "page_errors": [],
        }),
        ok=True,
    )

    assert flow.phase is ActionWorkflowPhase.CAPTURE
    assert flow.runtime_ready is True
    assert flow.runtime_blockers == ()
    assert flow.degraded_runtime_warnings
    assert "unaffected states" in flow.directive()


def test_functional_browser_task_stays_strict_when_optional_api_is_down() -> None:
    flow = ActionExecutionCoordinatorV1.build(
        ("read", "run", "preview"),
        screenshot_count=0,
        require_browser=True,
        require_visual_review=False,
        require_output=True,
    )
    flow.listed_files = ("package.json",)
    flow.read_files.add("package.json")
    flow.runtime_ready = True
    flow.runtime_url = "http://127.0.0.1:3000"
    flow.browser_opened = True
    flow.observe(
        "browser_inspect",
        {"browser_session_id": "browser-1"},
        json.dumps({
            "browser_session_id": "browser-1",
            "browser_opened": True,
            "text": "MediCare Home Doctors Login Register",
            "interaction_targets": [],
            "network_errors": [
                "GET http://localhost:7000/api/doctors: net::ERR_CONNECTION_REFUSED"
            ],
            "console_errors": [],
            "page_errors": [],
        }),
        ok=True,
    )

    assert flow.runtime_ready is False
    assert flow.runtime_blockers
    assert flow.phase is ActionWorkflowPhase.START_RUNTIME


def test_page_crash_reloads_verified_app_instead_of_restarting_runtime() -> None:
    flow = coordinator()
    flow.listed_files = ("client/package.json",)
    flow.read_files.add("client/package.json")
    flow.runtime_ready = True
    flow.runtime_url = "http://localhost:3000/"
    flow.browser_opened = True
    flow.browser_inspected = True
    flow.browser_session_id = "browser-1"

    flow.observe(
        "browser_act",
        {"browser_session_id": "browser-1"},
        json.dumps({
            "browser_session_id": "browser-1",
            "browser_opened": True,
            "page_errors": ["Objects are not valid as a React child"],
            "network_errors": [],
            "console_errors": [],
        }),
        ok=True,
    )

    assert flow.runtime_ready is True
    assert flow.phase is ActionWorkflowPhase.INSPECT_BROWSER
    repaired, reason = flow.rewrite_call(
        "browser_inspect", {"browser_session_id": "browser-1"}
    )
    assert repaired["url"] == "http://localhost:3000/"
    assert "reloaded" in reason
