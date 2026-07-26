from __future__ import annotations

import tempfile
import unittest

from agent.diagnostics import (
    audit_agent_readiness,
    benchmark_agent_readiness,
    probe_ollama_model_live,
    probe_ollama_orchestration_delta_live,
    probe_ollama_html_microtask_live,
    readiness_report_benchmark_payload,
    record_agent_readiness_report,
)
from agent.store import StateStore


class AgentReadinessAuditTests(unittest.TestCase):
    def test_readiness_audit_passes_when_required_gpu_is_declared_available(self) -> None:
        report = audit_agent_readiness(
            require_gpu=True,
            environ={
                "AGENT_GPU_AVAILABLE": "1",
                "AGENT_GPU_NAME": "RTX Test",
                "AGENT_GPU_DRIVER": "555.1",
            },
        )

        self.assertTrue(report.passed, report.to_dict())
        self.assertTrue(report.gpu.gpu_available)
        names = {item.name for item in report.checks}
        self.assertTrue(
            {
                "code_understanding",
                "retrieval",
                "reasoning_harness",
                "evaluation_benchmarks",
                "swarm_intelligence",
                "communication_protocol",
                "knowledge_and_learning",
                "project_memory_confidence",
                "ultra_evaluation_learning",
                "required_local_gpu",
            }
            <= names
        )

    def test_readiness_audit_fails_closed_when_gpu_is_required_but_unavailable(self) -> None:
        report = audit_agent_readiness(
            require_gpu=True,
            environ={"AGENT_GPU_AVAILABLE": "0"},
        )

        self.assertFalse(report.passed)
        failed = {item.name: item for item in report.failed_checks}
        self.assertIn("required_local_gpu", failed)
        self.assertIn("usable_local_gpu", failed["required_local_gpu"].missing)

    def test_readiness_report_converts_to_benchmark_payload(self) -> None:
        report = audit_agent_readiness(
            require_gpu=True,
            environ={
                "AGENT_GPU_AVAILABLE": "1",
                "AGENT_GPU_NAME": "RTX Test",
                "AGENT_GPU_DRIVER": "555.1",
            },
        )

        payload = readiness_report_benchmark_payload(report)

        self.assertEqual(payload["result"], "passed")
        self.assertIsNone(payload["blocker"])
        self.assertEqual(payload["metrics"]["failed_checks"], 0)
        self.assertEqual(payload["metrics"]["gpu_available"], 1)
        self.assertEqual(payload["scores"]["all_passed"], 1.0)
        self.assertEqual(payload["scores"]["pass_ratio"], 1.0)
        self.assertEqual(payload["inputs"]["failed_checks"], ())

    def test_readiness_report_records_passed_benchmark_run(self) -> None:
        report = audit_agent_readiness(
            require_gpu=True,
            environ={
                "AGENT_GPU_AVAILABLE": "1",
                "AGENT_GPU_NAME": "RTX Test",
                "AGENT_GPU_DRIVER": "555.1",
            },
        )
        with tempfile.TemporaryDirectory() as directory, StateStore(directory) as store:
            recorded = record_agent_readiness_report(
                store,
                report,
                scenario_name="structural",
                artifact_refs=("doctor://structural",),
            )
            rows = store.list_benchmark_results(
                suite_name="agent-readiness",
                scenario_name="structural",
            )

        self.assertEqual(recorded["result"], "passed")
        self.assertEqual(recorded["artifact_refs"], ["doctor://structural"])
        self.assertEqual(rows[0]["id"], recorded["id"])
        self.assertEqual(rows[0]["metrics"]["failed_checks"], 0)
        self.assertEqual(rows[0]["scores"]["all_passed"], 1.0)
        self.assertEqual(rows[0]["inputs"]["require_gpu"], True)
        self.assertEqual(rows[0]["inputs"]["gpu"]["devices"][0]["name"], "RTX Test")

    def test_readiness_report_records_failed_benchmark_run_with_blocker(self) -> None:
        report = audit_agent_readiness(
            require_gpu=True,
            environ={"AGENT_GPU_AVAILABLE": "0"},
        )
        with tempfile.TemporaryDirectory() as directory, StateStore(directory) as store:
            recorded = record_agent_readiness_report(
                store,
                report,
                scenario_name="gpu-required",
            )
            rows = store.list_benchmark_results(
                suite_name="agent-readiness",
                scenario_name="gpu-required",
            )

        self.assertEqual(recorded["result"], "failed")
        self.assertIn("required_local_gpu", recorded["blocker"])
        self.assertEqual(rows[0]["metrics"]["failed_checks"], 1)
        self.assertEqual(rows[0]["metrics"]["gpu_available"], 0)
        self.assertEqual(rows[0]["scores"]["all_passed"], 0.0)
        self.assertLess(rows[0]["scores"]["pass_ratio"], 1.0)
        failed = rows[0]["inputs"]["failed_checks"]
        self.assertEqual(failed[0]["name"], "required_local_gpu")

    def test_behavioral_readiness_benchmark_exercises_agent_architecture(self) -> None:
        report = benchmark_agent_readiness(
            require_gpu=False,
            environ={"AGENT_GPU_AVAILABLE": "0"},
        )

        self.assertTrue(report.passed, report.to_dict())
        names = {item.name for item in report.checks}
        self.assertEqual(
            names,
            {
                "behavioral_code_retrieval",
                "behavioral_reasoning",
                "behavioral_swarm_consensus",
                "behavioral_learning_evaluation",
            },
        )

    def test_behavioral_readiness_benchmark_fails_closed_for_required_gpu(self) -> None:
        report = benchmark_agent_readiness(
            require_gpu=True,
            environ={"AGENT_GPU_AVAILABLE": "0"},
        )

        self.assertFalse(report.passed)
        failed = {item.name: item for item in report.failed_checks}
        self.assertIn("behavioral_required_local_gpu", failed)
        self.assertIn("usable_local_gpu", failed["behavioral_required_local_gpu"].missing)

    def test_live_ollama_probe_accepts_valid_structured_json(self) -> None:
        def fake_http(method, url, *, payload=None, headers=None, timeout=0):
            self.assertEqual(method, "POST")
            self.assertTrue(url.endswith("/api/chat"))
            self.assertEqual(payload["model"], "gemma4:e4b")
            self.assertIs(payload["think"], False)
            self.assertEqual(payload["options"]["num_predict"], 192)
            return {
                "message": {
                    "content": '{"ok": true, "model": "gemma4:e4b", "verification": "valid"}'
                },
                "prompt_eval_count": 11,
                "eval_count": 7,
            }

        report = probe_ollama_model_live(
            "gemma4:e4b",
            require_gpu=True,
            environ={"AGENT_GPU_AVAILABLE": "1", "AGENT_GPU_NAME": "RTX Test"},
            http_json=fake_http,
        )

        self.assertTrue(report.passed, report.to_dict())
        self.assertIn("live_ollama_structured_json", {item.name for item in report.checks})
        self.assertIn("live_required_local_gpu", {item.name for item in report.checks})

    def test_live_ollama_probe_fails_on_invalid_model_json(self) -> None:
        def fake_http(_method, _url, *, payload=None, headers=None, timeout=0):
            return {"message": {"content": "not json"}}

        report = probe_ollama_model_live(
            "gemma4:e4b",
            environ={"AGENT_GPU_AVAILABLE": "1", "AGENT_GPU_NAME": "RTX Test"},
            http_json=fake_http,
        )

        self.assertFalse(report.passed)
        failed = {item.name: item for item in report.failed_checks}
        self.assertIn("live_ollama_structured_json", failed)

    def test_live_ollama_delta_records_orchestration_improvement(self) -> None:
        calls = []

        def fake_http(_method, _url, *, payload=None, headers=None, timeout=0):
            calls.append(dict(payload))
            if payload.get("think") is False:
                return {
                    "message": {
                        "content": '{"ok": true, "model": "gemma4:e4b", "verification": "controlled"}'
                    },
                    "prompt_eval_count": 12,
                    "eval_count": 8,
                }
            return {
                "message": {"content": "", "thinking": "spent all tokens thinking"},
                "done_reason": "length",
            }

        report = probe_ollama_orchestration_delta_live(
            "gemma4:e4b",
            require_gpu=True,
            environ={"AGENT_GPU_AVAILABLE": "1", "AGENT_GPU_NAME": "RTX Test"},
            http_json=fake_http,
        )

        self.assertTrue(report.passed, report.to_dict())
        delta = next(item for item in report.checks if item.name == "live_ollama_orchestration_delta")
        self.assertIn("raw_passed=False", delta.evidence)
        self.assertIn("controlled_passed=True", delta.evidence)
        self.assertIn("orchestration_improved=True", delta.evidence)
        self.assertNotIn("think", calls[0])
        self.assertIs(calls[1]["think"], False)

    def test_live_ollama_delta_fails_when_controlled_request_fails(self) -> None:
        def fake_http(_method, _url, *, payload=None, headers=None, timeout=0):
            return {"message": {"content": "not json"}}

        report = probe_ollama_orchestration_delta_live(
            "gemma4:e4b",
            environ={"AGENT_GPU_AVAILABLE": "1", "AGENT_GPU_NAME": "RTX Test"},
            http_json=fake_http,
        )

        self.assertFalse(report.passed)
        failed = {item.name: item for item in report.failed_checks}
        self.assertIn("live_ollama_orchestration_delta", failed)
        self.assertIn("controlled_structured_json", failed["live_ollama_orchestration_delta"].missing)

    def test_live_html_microtask_executes_benchmark_and_records_quality_gap(self) -> None:
        def fake_http(_method, _url, *, payload=None, headers=None, timeout=0):
            self.assertIs(payload["think"], False)
            self.assertEqual(payload["options"]["num_predict"], 4096)
            return {
                "message": {
                    "content": '{"ok": true, "model": "gemma4:e4b", "html": "<!doctype html><html><body><h1>Demo</h1></body></html>"}'
                }
            }

        report = probe_ollama_html_microtask_live(
            "gemma4:e4b",
            require_gpu=True,
            environ={"AGENT_GPU_AVAILABLE": "1", "AGENT_GPU_NAME": "RTX Test"},
            http_json=fake_http,
        )

        self.assertTrue(report.passed, report.to_dict())
        check = next(item for item in report.checks if item.name == "live_ollama_html_microtask")
        self.assertIn("benchmark_passed=False", check.evidence)
        self.assertTrue(any(item.startswith("finding:") for item in check.evidence))

    def test_live_html_microtask_can_be_quality_required(self) -> None:
        def fake_http(_method, _url, *, payload=None, headers=None, timeout=0):
            return {
                "message": {
                    "content": '{"ok": true, "model": "gemma4:e4b", "html": "<!doctype html><html><body><h1>Demo</h1></body></html>"}'
                }
            }

        report = probe_ollama_html_microtask_live(
            "gemma4:e4b",
            require_quality=True,
            environ={"AGENT_GPU_AVAILABLE": "1", "AGENT_GPU_NAME": "RTX Test"},
            http_json=fake_http,
        )

        self.assertFalse(report.passed)
        failed = {item.name: item for item in report.failed_checks}
        self.assertIn("html_quality_threshold", failed["live_ollama_html_microtask"].missing)

    def test_live_html_microtask_refinement_can_pass_quality_gate(self) -> None:
        weak = "<!doctype html><html><body><h1>Demo</h1></body></html>"
        rich = """
<!doctype html><html><head><title>Neon Arena</title><meta name='viewport' content='width=device-width,initial-scale=1'>
<style>body{margin:0;background:radial-gradient(circle,#123,#001)}canvas{display:block}.hud{color:white;background:linear-gradient(#224,#112)}</style></head>
<body><canvas id='game' role='img' aria-label='3D arena'></canvas><div class='hud'>score health level</div>
<script>
const THREE={Scene:class{constructor(){this.items=[]}add(x){this.items.push(x)}},Fog:class{},PerspectiveCamera:class{constructor(){this.aspect=1}updateProjectionMatrix(){}},WebGLRenderer:class{constructor(){this.shadowMap={}}setSize(){}render(){}},AmbientLight:class{},PointLight:class{},DirectionalLight:class{},SpotLight:class{},MeshStandardMaterial:class{},BoxGeometry:class{},SphereGeometry:class{},PlaneGeometry:class{},CylinderGeometry:class{},BufferGeometry:class{},Mesh:class{constructor(){this.position={distanceTo(){return 1}};this.rotation={};this.castShadow=true}},InstancedMesh:class{}};
const scene=new THREE.Scene(); scene.fog=new THREE.Fog(); const camera=new THREE.PerspectiveCamera(); const renderer=new THREE.WebGLRenderer({canvas:game}); renderer.setSize(innerWidth,innerHeight);
scene.add(new THREE.AmbientLight()); scene.add(new THREE.PointLight()); scene.add(new THREE.DirectionalLight()); scene.add(new THREE.SpotLight());
const mat=new THREE.MeshStandardMaterial({emissive:1,roughness:.2,metalness:.8,shadow:true}); const enemies=[], projectiles=[], particles=[], trail=[], bloom=[];
for(let i=0;i<20;i++){scene.add(new THREE.Mesh(new THREE.BoxGeometry(),mat)); scene.add(new THREE.Mesh(new THREE.SphereGeometry(),mat)); scene.add(new THREE.Mesh(new THREE.PlaneGeometry(),mat));}
let score=0,health=100,level=1,velocity=0; addEventListener('keydown',()=>velocity=1); addEventListener('keyup',()=>velocity=0);
function collision(a,b){return a.position.distanceTo(b.position)<2} function hit(){score++;health--;level++;projectiles.push({});enemies.push({});particles.push({});trail.push({});}
function animate(){requestAnimationFrame(animate); hit(); renderer.render(scene,camera)} addEventListener('resize',()=>{camera.aspect=innerWidth/innerHeight;camera.updateProjectionMatrix();renderer.setSize(innerWidth,innerHeight)}); animate();
</script></body></html>
"""
        responses = [weak, rich]

        def fake_http(_method, _url, *, payload=None, headers=None, timeout=0):
            html = responses.pop(0)
            return {
                "message": {
                    "content": '{"ok": true, "model": "gemma4:e4b", "html": ' + __import__("json").dumps(html) + "}"
                }
            }

        report = probe_ollama_html_microtask_live(
            "gemma4:e4b",
            require_quality=True,
            refine_attempts=1,
            environ={"AGENT_GPU_AVAILABLE": "1", "AGENT_GPU_NAME": "RTX Test"},
            http_json=fake_http,
        )

        self.assertTrue(report.passed, report.to_dict())
        check = next(item for item in report.checks if item.name == "live_ollama_html_microtask")
        self.assertIn("attempts=2", check.evidence)
        self.assertIn("benchmark_passed=True", check.evidence)
        self.assertIn("refinement_improved=True", check.evidence)


if __name__ == "__main__":
    unittest.main()
