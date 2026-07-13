import io
import os
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

from agent.convergence import (
    ConvergenceWatchdog, EvaluationConfidence, EvaluatorCapabilityProfile,
    QualityDimensionV1, QualityEvaluationV1, QualityRubricV1, QualityScoreV1,
    QualityTargetV1, evaluation_passes,
)
from agent.evaluation import (
    RetrievalBenchmarkCase,
    record_repository_retrieval_benchmark,
    record_single_file_3d_html_benchmark,
    run_repository_retrieval_benchmark,
    run_single_file_3d_html_benchmark,
)
from agent.local_provider import (ModelCapabilityProfile, OllamaRequestCompiler, ProviderFailureKind,
    ProviderRequestError, extract_first_json_object, normalize_action_proposal)
from agent.providers.ollama_provider import OllamaProvider
from agent.reasoning import evaluate_reasoning_artifact, reasoning_debate_protocol_for
from agent.repository_index import HashingEmbeddingProvider, RepositoryIndex
from agent.run_context import RunContextV1
from agent.store import StateStore
from agent.workflow import SessionMode


class QualityConvergenceTests(unittest.TestCase):
    def test_fresh_evaluation_meets_explicit_target(self):
        dimension = QualityDimensionV1("functional", "works", minimum_score=.8)
        target = QualityTargetV1("build", ("index.html",), QualityRubricV1((dimension,)), hard_gates=("tests",))
        evaluation = QualityEvaluationV1(target.id, {"index.html": "abc"},
            (QualityScoreV1("functional", .9, True, ("test-1",), confidence=EvaluationConfidence.HIGH),),
            {"tests": True}, EvaluatorCapabilityProfile("pytest", deterministic=True), .9)
        self.assertTrue(evaluation_passes(target, evaluation))

    def test_watchdog_requires_different_approach_after_three_failures(self):
        watchdog = ConvergenceWatchdog()
        self.assertEqual(watchdog.record("same", progress=False), "retry")
        self.assertEqual(watchdog.record("same", progress=False), "retry")
        self.assertEqual(watchdog.record("same", progress=False), "replan")

    def test_mode_transition_preserves_run_identity(self):
        context = RunContextV1("workspace", "hash", "objective", "objective")
        changed = context.transition(SessionMode.PLAN).transition(SessionMode.GOAL)
        self.assertEqual(changed.run_id, context.run_id)
        self.assertEqual(changed.original_objective, "objective")

    def test_html_index_retrieves_catapult_without_losing_structure(self):
        html = '<div id="castle"></div><style>@keyframes catapultLaunch {to{transform:rotate(20deg)}} @media(max-width:600px){#castle{width:90%}}</style><script>function fireArrow(){}; function launchCatapult(){}</script>'
        with tempfile.TemporaryDirectory() as root:
            Path(root, "siege.html").write_text(html, encoding="utf-8")
            index = RepositoryIndex(root); index.update("siege.html")
            matches = index.search("catapult animation")
            self.assertTrue(any("catapult" in item.name.casefold() or "catapult" in item.text.casefold() for item in matches))
            self.assertTrue(any(item.kind == "keyframe" for item in matches))

    def test_mixed_html_index_separates_visual_runtime_responsive_and_accessibility_components(self):
        html = '''<main id="castle" aria-label="fortress"><svg><g id="siege-tower"><path id="gate" d="M0 0"/></g></svg>
        <canvas id="battle"></canvas><style>:root{--torch-color:orange}.castle{filter:brightness(.8)}
        @keyframes ramStrike{to{transform:translateX(2px)}} @media(max-width:600px){.castle{width:90%}}</style>
        <script>const arrowProjectiles=[]; const collisionState={}; function fireArrow(){} function launchCatapult(){}
        const ctx=document.querySelector('canvas').getContext('2d'); requestAnimationFrame(renderBattle);</script></main>'''
        with tempfile.TemporaryDirectory() as root:
            Path(root, "siege.html").write_text(html, encoding="utf-8")
            index = RepositoryIndex(root); entries = index.update("siege.html")
            kinds = {entry.kind for entry in entries}
            self.assertTrue({"dom", "svg_element", "css_variable", "keyframe", "responsive", "js_function",
                             "canvas_setup", "state_variable", "animation_loop", "projectile_system", "accessibility"} <= kinds)
            self.assertTrue(index.search("castle lighting colors"))
            self.assertTrue(index.search("arrow projectile logic"))
            self.assertTrue(index.search("siege tower movement"))

    def test_python_ast_index_builds_import_call_and_ownership_graphs(self):
        code = '''
import json
from pathlib import Path

class Loader:
    def load(self, raw):
        data = json.loads(raw)
        return normalize(data)

def normalize(value):
    return Path(str(value)).name

def handle(raw):
    loader = Loader()
    return loader.load(raw)
'''
        with tempfile.TemporaryDirectory() as root:
            Path(root, "app.py").write_text(code, encoding="utf-8")
            index = RepositoryIndex(root)
            entries = index.update("app.py")
            kinds = {entry.kind for entry in entries}
            self.assertTrue({"py_import", "py_class", "py_method", "py_function", "py_call"} <= kinds)
            self.assertEqual(index.dependency_graph()["app.py"], ("json", "pathlib.Path"))
            self.assertIn("Loader.load", index.ownership_graph()["Loader"])
            self.assertIn("json.loads", index.call_graph()["Loader.load"])
            self.assertIn("Loader", index.call_graph()["handle"])
            graph_hits = index.search_graph("who calls json.loads", relation_kinds=("call",))
            self.assertEqual(graph_hits[0].source, "Loader.load")
            hybrid = index.hybrid_search("json.loads caller", kinds=("py_method", "py_function"))
            self.assertEqual(hybrid[0].name, "Loader.load")

    def test_cross_file_python_index_resolves_imported_callers_and_callees(self):
        with tempfile.TemporaryDirectory() as root:
            services = Path(root, "services")
            api = Path(root, "api")
            services.mkdir()
            api.mkdir()
            Path(services, "__init__.py").write_text("", encoding="utf-8")
            Path(api, "__init__.py").write_text("", encoding="utf-8")
            Path(services, "auth.py").write_text(
                '''
class AuthService:
    def issue_token(self, user_id):
        return f"token:{user_id}"
''',
                encoding="utf-8",
            )
            Path(api, "routes.py").write_text(
                '''
from services.auth import AuthService

def login(user_id):
    service = AuthService()
    return service.issue_token(user_id)
''',
                encoding="utf-8",
            )

            index = RepositoryIndex(root)
            index.update_all()

            self.assertEqual(index.resolved_dependency_graph()["api/routes.py"], ("services/auth.py",))
            calls = index.resolved_call_graph()
            self.assertIn("services.auth.AuthService", calls["api.routes.login"])
            self.assertIn("services.auth.AuthService.issue_token", calls["api.routes.login"])
            self.assertEqual(index.callers_of("services.auth.AuthService.issue_token"), ("api.routes.login",))
            self.assertIn("services.auth.AuthService.issue_token", index.callees_of("api.routes.login"))
            symbols = index.symbol_index()
            self.assertIn("services.auth.AuthService.issue_token", symbols)

    def test_repository_context_slice_focuses_large_repo_with_graph_neighbors(self):
        with tempfile.TemporaryDirectory() as root:
            services = Path(root, "services")
            api = Path(root, "api")
            noise = Path(root, "noise")
            services.mkdir()
            api.mkdir()
            noise.mkdir()
            Path(services, "__init__.py").write_text("", encoding="utf-8")
            Path(api, "__init__.py").write_text("", encoding="utf-8")
            Path(services, "auth.py").write_text(
                '''
class AuthService:
    def issue_token(self, user_id):
        audit = {"event": "issue_token", "user": user_id}
        return f"token:{user_id}:{audit['event']}"
''',
                encoding="utf-8",
            )
            Path(api, "routes.py").write_text(
                '''
from services.auth import AuthService

def login(user_id):
    service = AuthService()
    return service.issue_token(user_id)
''',
                encoding="utf-8",
            )
            for index_num in range(40):
                Path(noise, f"module_{index_num}.py").write_text(
                    f"def unrelated_{index_num}():\n    return {index_num}\n",
                    encoding="utf-8",
                )

            index = RepositoryIndex(root)
            index.update_all()
            context = index.context_slice(
                "where is login token issued",
                kinds=("py_function", "py_method", "py_class"),
                max_entries=6,
                budget_chars=4_000,
            )

            names = {entry.name for entry in context.entries}
            self.assertIn("login", names)
            self.assertIn("AuthService.issue_token", names)
            self.assertLessEqual(context.size_chars, 4_000)
            self.assertLessEqual(len(context.entries), 6)
            self.assertIn("api.routes.login", context.callees)
            self.assertIn("services.auth.AuthService.issue_token", context.callees["api.routes.login"])
            self.assertEqual(context.dependencies["api/routes.py"], ("services/auth.py",))
            self.assertFalse(any(entry.path.startswith("noise/") for entry in context.entries))

    def test_repository_index_reuses_unchanged_files_and_refreshes_changed_files(self):
        with tempfile.TemporaryDirectory() as root:
            alpha = Path(root, "alpha.py")
            beta = Path(root, "beta.py")
            alpha.write_text("def alpha():\n    return 1\n", encoding="utf-8")
            beta.write_text("from alpha import alpha\n\ndef beta():\n    return alpha()\n", encoding="utf-8")
            index = RepositoryIndex(root)

            first = index.update_all()
            self.assertGreaterEqual(len(first), 4)
            self.assertEqual(index.last_update_stats["seen"], 2)
            self.assertEqual(index.last_update_stats["updated"], 2)
            self.assertEqual(index.last_update_stats["reused"], 0)

            second = index.update_all()
            self.assertEqual(len(second), len(first))
            self.assertEqual(index.last_update_stats["updated"], 0)
            self.assertEqual(index.last_update_stats["reused"], 2)

            alpha.write_text("def alpha():\n    return 2\n\ndef gamma():\n    return alpha()\n", encoding="utf-8")
            next_time = alpha.stat().st_mtime_ns + 10_000_000
            os.utime(alpha, ns=(next_time, next_time))
            third = index.update_all()
            self.assertGreater(len(third), len(second))
            self.assertEqual(index.last_update_stats["updated"], 1)
            self.assertEqual(index.last_update_stats["reused"], 1)
            self.assertEqual(index.hybrid_search("gamma", kinds=("py_function",))[0].name, "gamma")

            beta.unlink()
            index.update_all()
            self.assertEqual(index.last_update_stats["removed"], 1)
            self.assertNotIn("beta.py", index.entries)
            self.assertNotIn("beta.py", index.dependency_graph())
            self.assertFalse(any(item.path == "beta.py" for item in index.hybrid_search("beta")))

    def test_repository_index_partial_scan_does_not_purge_unseen_cached_files(self):
        with tempfile.TemporaryDirectory() as root:
            for name in ("a.py", "b.py", "c.py"):
                Path(root, name).write_text(f"def {name[0]}_func():\n    return '{name}'\n", encoding="utf-8")
            index = RepositoryIndex(root)
            index.update_all()
            self.assertEqual(set(index.entries), {"a.py", "b.py", "c.py"})

            partial = index.update_all(max_files=1)

            self.assertTrue(partial)
            self.assertEqual(index.last_update_stats["seen"], 1)
            self.assertEqual(index.last_update_stats["removed"], 0)
            self.assertEqual(set(index.entries), {"a.py", "b.py", "c.py"})

    def test_repository_index_persistent_cache_reuses_across_instances(self):
        class CountingEmbeddingProvider:
            dimensions = 3

            def __init__(self):
                self.calls = 0

            def embed(self, text: str):
                self.calls += 1
                lowered = text.casefold()
                if "alpha" in lowered:
                    return (1.0, 0.0, 0.0)
                if "beta" in lowered:
                    return (0.0, 1.0, 0.0)
                return (0.0, 0.0, 1.0)

        with tempfile.TemporaryDirectory() as root:
            Path(root, "alpha.py").write_text("def alpha():\n    return 'alpha'\n", encoding="utf-8")
            Path(root, "beta.py").write_text("def beta():\n    return 'beta'\n", encoding="utf-8")
            cache_path = Path(root, ".coding-agent", "repository-index-v1.json")
            first_provider = CountingEmbeddingProvider()
            first = RepositoryIndex(root, embedding_provider=first_provider, cache_path=cache_path)
            first.update_all()
            self.assertTrue(cache_path.is_file())
            self.assertGreater(first_provider.calls, 0)

            second_provider = CountingEmbeddingProvider()
            second = RepositoryIndex(root, embedding_provider=second_provider, cache_path=cache_path)
            self.assertEqual(second.last_update_stats["loaded"], 2)
            self.assertEqual(second_provider.calls, 0)
            reused = second.update_all()

            self.assertTrue(reused)
            self.assertEqual(second.last_update_stats["updated"], 0)
            self.assertEqual(second.last_update_stats["reused"], 2)
            self.assertEqual(second_provider.calls, 0)
            self.assertEqual(second.hybrid_search("alpha", kinds=("py_function",))[0].name, "alpha")

    def test_repository_index_persistent_cache_invalidates_changed_file(self):
        with tempfile.TemporaryDirectory() as root:
            target = Path(root, "alpha.py")
            target.write_text("def alpha():\n    return 'alpha'\n", encoding="utf-8")
            cache_path = Path(root, ".coding-agent", "repository-index-v1.json")
            first = RepositoryIndex(root, cache_path=cache_path)
            first.update_all()

            second = RepositoryIndex(root, cache_path=cache_path)
            target.write_text("def alpha():\n    return 'alpha'\n\ndef delta():\n    return alpha()\n", encoding="utf-8")
            next_time = target.stat().st_mtime_ns + 10_000_000
            os.utime(target, ns=(next_time, next_time))
            second.update_all()

            self.assertEqual(second.last_update_stats["updated"], 1)
            self.assertEqual(second.hybrid_search("delta", kinds=("py_function",))[0].name, "delta")

    def test_semantic_retrieval_finds_code_when_query_uses_different_words(self):
        code = '''
import json

def parse_payload(raw):
    return json.loads(raw)

def paint_scene(canvas):
    requestAnimationFrame(canvas.draw)
'''
        with tempfile.TemporaryDirectory() as root:
            Path(root, "app.py").write_text(code, encoding="utf-8")
            index = RepositoryIndex(root)
            index.update("app.py")

            deserialize = index.semantic_search(
                "deserialize json request body",
                kinds=("py_function",),
            )
            self.assertEqual(deserialize[0].name, "parse_payload")

            visual = index.hybrid_search("visual rendering loop", kinds=("py_function",))
            self.assertEqual(visual[0].name, "paint_scene")

    def test_embedding_retrieval_is_explicit_and_pluggable(self):
        class KeywordEmbeddingProvider:
            def embed(self, text: str):
                lowered = text.casefold()
                if any(term in lowered for term in ("auth", "authenticate", "token")):
                    return (1.0, 0.0, 0.0)
                if any(term in lowered for term in ("paint", "render", "canvas")):
                    return (0.0, 1.0, 0.0)
                return (0.0, 0.0, 1.0)

        code = '''
def issue_token(user_id):
    return f"token:{user_id}"

def paint_scene(canvas):
    return canvas.draw()
'''
        with tempfile.TemporaryDirectory() as root:
            Path(root, "app.py").write_text(code, encoding="utf-8")
            index = RepositoryIndex(root, embedding_provider=KeywordEmbeddingProvider())
            index.update("app.py")

            auth_hits = index.embedding_search("authenticate user session", kinds=("py_function",))
            self.assertEqual(auth_hits[0].name, "issue_token")
            scored = index.search_with_scores("authenticate user session", kinds=("py_function",))
            self.assertEqual(scored[0].entry.name, "issue_token")
            self.assertIn("embedding", scored[0].channels)

    def test_default_hashing_embedding_provider_supports_offline_embedding_search(self):
        code = '''
def enqueue_retry(job):
    return {"retry": job}
'''
        with tempfile.TemporaryDirectory() as root:
            Path(root, "worker.py").write_text(code, encoding="utf-8")
            index = RepositoryIndex(root, embedding_provider=HashingEmbeddingProvider(dimensions=32))
            index.update_all()

            hits = index.embedding_search("retry queue worker", kinds=("py_function",))
            self.assertEqual(hits[0].name, "enqueue_retry")

    def test_repository_retrieval_benchmark_reports_accuracy_and_rank(self):
        code = '''
import json

def parse_payload(raw):
    return json.loads(raw)

def paint_scene(canvas):
    requestAnimationFrame(canvas.draw)
'''
        with tempfile.TemporaryDirectory() as root:
            Path(root, "app.py").write_text(code, encoding="utf-8")
            index = RepositoryIndex(root)
            index.update("app.py")

            result = run_repository_retrieval_benchmark(
                index,
                (
                    RetrievalBenchmarkCase(
                        "deserialize request body",
                        ("parse_payload",),
                        kinds=("py_function",),
                    ),
                    RetrievalBenchmarkCase(
                        "graphics animation frame",
                        ("paint_scene",),
                        kinds=("py_function",),
                    ),
                ),
            )

            self.assertTrue(result.passed)
            self.assertEqual(result.metrics["accuracy_at_k"], 1.0)
            self.assertGreaterEqual(result.metrics["mean_reciprocal_rank"], 0.5)

    def test_repository_retrieval_benchmark_records_first_class_metrics(self):
        with tempfile.TemporaryDirectory() as root:
            Path(root, "worker.py").write_text(
                '''
def enqueue_retry(job):
    return {"retry": job}
''',
                encoding="utf-8",
            )
            index = RepositoryIndex(root)
            index.update_all()
            store = StateStore(Path(root))
            try:
                recorded = record_repository_retrieval_benchmark(
                    store,
                    index,
                    (
                        RetrievalBenchmarkCase(
                            "retry queue worker",
                            ("enqueue_retry",),
                            kinds=("py_function",),
                        ),
                    ),
                    provider="ollama",
                    model="gemma4",
                    artifact_refs=("workspace:worker.py",),
                )

                self.assertEqual(recorded["suite_name"], "repository-retrieval")
                self.assertEqual(recorded["scenario_name"], "hybrid-search")
                self.assertEqual(recorded["result"], "passed")
                self.assertEqual(recorded["scores"]["accuracy_at_k"], 1.0)
                self.assertEqual(recorded["artifact_refs"], ["workspace:worker.py"])
                self.assertEqual(recorded["inputs"]["cases"][0]["top_hits"][0]["name"], "enqueue_retry")
            finally:
                store.close()

    def test_reasoning_debate_protocol_scores_external_artifacts_without_cot(self):
        protocol = reasoning_debate_protocol_for("tester", "test", {"contract": {"objective": "verify"}})
        missing = evaluate_reasoning_artifact({}, protocol)
        self.assertFalse(missing.passed)
        self.assertIn("claim", missing.missing_fields)

        good = evaluate_reasoning_artifact(
            {
                "claim": "The browser runtime passes.",
                "supporting_evidence": [{"test": "preview_html", "passed": True}],
                "counterarguments": ["Console errors could still appear on another viewport."],
                "rejected_alternatives": ["Trusting model prose without browser evidence."],
                "verification_plan": ["Run preview_html and inspect console errors."],
                "reasoning_graph": {
                    "nodes": [
                        {
                            "id": "browser-pass",
                            "type": "verification",
                            "summary": "Browser runtime evidence supports the pass claim.",
                            "status": "verified",
                            "evidence_refs": ["preview_html"],
                        },
                        {
                            "id": "prose-only",
                            "type": "option",
                            "summary": "Prose-only acceptance was considered insufficient.",
                            "status": "rejected",
                            "evidence_refs": [],
                        },
                    ],
                    "edges": [
                        {"from": "browser-pass", "to": "prose-only", "relation": "rejects"}
                    ],
                },
            },
            protocol,
        )
        self.assertTrue(good.passed)
        self.assertGreaterEqual(good.score, 0.8)

    def test_reasoning_graph_blocks_shallow_external_artifacts(self):
        protocol = reasoning_debate_protocol_for("coder", "implement", {"contract": {"objective": "build"}})
        shallow = evaluate_reasoning_artifact(
            {
                "claim": "The implementation works.",
                "supporting_evidence": ["I checked it."],
                "counterarguments": ["It might fail."],
                "rejected_alternatives": ["Other code."],
                "verification_plan": ["Run tests."],
            },
            protocol,
        )

        self.assertFalse(shallow.passed)
        self.assertIn("reasoning_graph", shallow.weak_fields)

    def test_single_file_3d_html_benchmark_rejects_static_placeholder(self):
        weak = "<!doctype html><html><title>Game</title><body><h1>3D Game</h1><p>Coming soon</p></body></html>"
        result = run_single_file_3d_html_benchmark(weak)

        self.assertFalse(result.passed)
        self.assertLess(result.scores["overall"], 0.5)
        self.assertIn("3D/WebGL renderer", "; ".join(result.findings))
        self.assertIn("Gameplay", "; ".join(result.findings))

    def test_single_file_3d_html_benchmark_scores_rich_game_candidate(self):
        rich = """
<!doctype html><html><head><title>Neon Rift Arena</title><meta name="viewport" content="width=device-width,initial-scale=1">
<style>body{margin:0;background:radial-gradient(circle,#102,#001);overflow:hidden}.hud{position:fixed;color:white}</style></head>
<body><canvas id="game" aria-label="Neon 3D arena" role="img"></canvas><div class="hud">score health level</div>
<script>
const scene = new THREE.Scene(); scene.fog = new THREE.Fog(0x020014, 10, 90);
const camera = new THREE.PerspectiveCamera(70, innerWidth/innerHeight, .1, 1000);
const renderer = new THREE.WebGLRenderer({canvas:document.getElementById('game'), antialias:true});
renderer.setSize(innerWidth, innerHeight); renderer.shadowMap.enabled = true;
scene.add(new THREE.AmbientLight(0x3344ff, .5)); scene.add(new THREE.PointLight(0xff44cc, 2));
const material = new THREE.MeshStandardMaterial({color:0x33ffee, emissive:0x112244, roughness:.25, metalness:.7});
for(let i=0;i<24;i++){ const mesh = new THREE.Mesh(new THREE.BoxGeometry(1,1,1), material); mesh.castShadow=true; scene.add(mesh); }
const enemies=[], projectiles=[]; let score=0, health=100, level=1, velocity={x:0,z:0};
addEventListener('keydown', e => { velocity.x = e.key === 'ArrowRight' ? 1 : velocity.x; });
addEventListener('keyup', e => { velocity.x = 0; });
function collision(a,b){ return a.position.distanceTo(b.position) < 1.2; }
function spawnEnemy(){ enemies.push(new THREE.Mesh(new THREE.SphereGeometry(.5), material)); }
function fireProjectile(){ projectiles.push({hit:false, velocity:2}); }
function animate(){ requestAnimationFrame(animate); enemies.forEach(e=>e.rotation.y+=.03); projectiles.forEach(p=>p.hit = p.hit || false); renderer.render(scene,camera); }
addEventListener('resize',()=>{camera.aspect=innerWidth/innerHeight;camera.updateProjectionMatrix();renderer.setSize(innerWidth,innerHeight);});
animate();
</script></body></html>
"""
        result = run_single_file_3d_html_benchmark(
            rich,
            preview={"verification": "passed", "console_errors": [], "page_errors": [], "network_errors": []},
        )

        self.assertTrue(result.passed)
        self.assertGreaterEqual(result.scores["overall"], 0.8)
        self.assertEqual(result.metrics["runtime_pass"], 1.0)

    def test_single_file_3d_html_benchmark_records_first_class_metrics(self):
        rich = """
<!doctype html><html><head><title>3D</title><meta name="viewport" content="width=device-width">
<style>body{background:linear-gradient(#102,#001)}.hud{filter:drop-shadow(0 0 8px cyan)}</style></head><body>
<canvas role="img" aria-label="arena"></canvas><div class="hud">score health</div><script>
const scene=new THREE.Scene(), camera=new THREE.PerspectiveCamera(), renderer=new THREE.WebGLRenderer();
scene.fog=new THREE.Fog(0x001122,10,80); scene.add(new THREE.AmbientLight()); scene.add(new THREE.PointLight());
const m=new THREE.MeshStandardMaterial({emissive:1,roughness:.2,metalness:.5});
for(let i=0;i<30;i++){scene.add(new THREE.Mesh(new THREE.BoxGeometry(),m))}
let score=0, health=5, enemies=[], projectiles=[], particles=[], trail=[], velocity=0, bloom=true;
addEventListener('keydown',()=>velocity=1); function collision(){return false}
function lerp(a,b,t){return a+(b-a)*t} function animate(){requestAnimationFrame(animate);renderer.render(scene,camera)} addEventListener('resize',()=>renderer.setSize(innerWidth,innerHeight)); animate();
</script></body></html>
"""
        with tempfile.TemporaryDirectory() as root:
            store = StateStore(Path(root))
            try:
                recorded = record_single_file_3d_html_benchmark(
                    store,
                    rich,
                    provider="ollama",
                    model="gemma4",
                    preview={"verification": "passed", "console_errors": [], "page_errors": [], "network_errors": []},
                )
                self.assertEqual(recorded["suite_name"], "weak-model-html")
                self.assertEqual(recorded["scenario_name"], "threejs-single-file")
                self.assertEqual(recorded["result"], "passed")
                self.assertGreaterEqual(recorded["scores"]["overall"], 0.8)
                self.assertEqual(recorded["artifact_refs"], ["workspace:index.html"])
            finally:
                store.close()

    def test_python_ast_index_reports_syntax_errors_as_searchable_entries(self):
        with tempfile.TemporaryDirectory() as root:
            Path(root, "broken.py").write_text("def nope(:\n    pass\n", encoding="utf-8")
            index = RepositoryIndex(root)
            entries = index.update("broken.py")
            self.assertEqual(entries[1].kind, "python_syntax_error")
            self.assertTrue(index.search("invalid syntax") or index.search("SyntaxError") or index.search("invalid"))

    def test_request_compiler_omits_unsupported_contracts(self):
        profile = ModelCapabilityProfile("weak", tool_call_support=False, structured_output_support=False)
        payload = OllamaRequestCompiler().compile(profile, messages=[{"role":"user","content":"x"}], tools=[{"name":"shell"}], structured=True)
        self.assertNotIn("tools", payload); self.assertNotIn("format", payload)

    def test_non_native_tool_action_is_extracted_from_bounded_prose(self):
        candidate = extract_first_json_object('proposal:\n```json\n{"tool":"read_file","arguments":{"path":"x.py"}}\n```')
        self.assertEqual(normalize_action_proposal(candidate), ("read_file", {"path": "x.py"}))

    def test_http_400_is_reachable_request_rejection(self):
        error = urllib.error.HTTPError("http://localhost:11434/api/chat", 400, "bad", {}, io.BytesIO(b'{"error":"unknown field tools; token=secret"}'))
        provider = OllamaProvider(model="weak")
        with patch("urllib.request.urlopen", side_effect=error):
            with self.assertRaises(ProviderRequestError) as raised:
                provider._post_json("/api/chat", {})
        diagnostic = raised.exception.diagnostic
        self.assertTrue(diagnostic.reachable)
        self.assertEqual(diagnostic.kind, ProviderFailureKind.UNSUPPORTED_PARAMETER)
        self.assertEqual(diagnostic.status_code, 400)
        self.assertNotIn("Could not reach", str(raised.exception))
        self.assertNotIn("secret", diagnostic.provider_message)

    def test_unsupported_thinking_http_400_identifies_safe_field_adaptation(self):
        error = urllib.error.HTTPError("http://localhost:11434/api/chat", 400, "bad", {}, io.BytesIO(b'{"error":"model does not support thinking"}'))
        provider = OllamaProvider(model="weak")
        with patch("urllib.request.urlopen", side_effect=error):
            with self.assertRaises(ProviderRequestError) as raised:
                provider._post_json("/api/chat", {"think":"medium"})
        self.assertTrue(raised.exception.diagnostic.reachable)
        self.assertEqual(raised.exception.diagnostic.kind, ProviderFailureKind.UNSUPPORTED_PARAMETER)
        self.assertEqual(raised.exception.diagnostic.incompatible_field, "think")


if __name__ == "__main__": unittest.main()
