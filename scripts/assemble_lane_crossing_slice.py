"""Assemble exact accepted component bytes into a playable Ultra vertical slice."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--road", required=True)
    parser.add_argument("--vehicle", required=True)
    parser.add_argument("--character", required=True)
    parser.add_argument("--specs", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    inputs = {name: Path(value).resolve(strict=True) for name, value in {
        "road": args.road, "vehicle": args.vehicle, "character": args.character,
    }.items()}
    source = {name: path.read_text(encoding="utf-8") for name, path in inputs.items()}
    specs_path = Path(args.specs).resolve(strict=True)
    specs = json.loads(specs_path.read_text(encoding="utf-8"))
    # A near-white accent has no semantic hierarchy on the light scene. Keep
    # the specialist choice unless it fails this deterministic contrast guard.
    accent_text = str(specs["presentation"].get("hud_accent") or "#ffd34e")
    try:
        rgb = tuple(int(accent_text[index:index + 2], 16) for index in (1, 3, 5))
    except (ValueError, IndexError):
        rgb = (255, 211, 78)
    if max(rgb) - min(rgb) < 36 or sum(rgb) / 3 > 235:
        specs["presentation"]["hud_accent"] = "#ffd34e"
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    template = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"><link rel="icon" href="data:,">
<title>Sunny Crossing — Ultra Slice</title>
<style>
:root{--accent:__ACCENT__;--ink:#17232b;--glass:rgba(17,31,40,.86)}*{box-sizing:border-box}html,body{margin:0;width:100%;height:100%;overflow:hidden;background:#17232b;color:white;font-family:Inter,ui-rounded,"Segoe UI",sans-serif}canvas{display:block;width:100%;height:100%;touch-action:none}.hud{position:fixed;inset:0;pointer-events:none;padding:max(18px,env(safe-area-inset-top)) max(18px,env(safe-area-inset-right)) max(18px,env(safe-area-inset-bottom)) max(18px,env(safe-area-inset-left));display:flex;justify-content:space-between;align-items:flex-start}.brand{font-weight:900;letter-spacing:.08em;text-transform:uppercase;text-shadow:0 2px 10px #0008}.brand small{display:block;font-size:10px;opacity:.72;letter-spacing:.2em}.score{font-variant-numeric:tabular-nums;background:var(--glass);border-left:4px solid var(--accent);padding:10px 15px;border-radius:4px 16px 16px 4px;box-shadow:0 10px 32px #0003}.score b{font-size:25px;color:var(--accent)}.score span{font-size:11px;opacity:.7;margin-left:10px}.center-note{position:fixed;left:50%;top:12%;transform:translateX(-50%);font-weight:800;letter-spacing:.08em;text-transform:uppercase;text-shadow:0 2px 12px #0009;pointer-events:none;transition:.25s}.overlay{position:fixed;inset:0;display:grid;place-items:center;background:rgba(10,20,27,.34);backdrop-filter:blur(3px);opacity:0;visibility:hidden;transition:.25s}.overlay.show{opacity:1;visibility:visible}.overlay section{text-align:center}.overlay h1{font-size:clamp(38px,8vw,82px);line-height:.9;margin:0 0 14px;text-transform:uppercase}.overlay p{opacity:.82}.overlay button{pointer-events:auto;border:0;border-radius:999px;padding:13px 24px;font-weight:900;background:var(--accent);color:var(--ink);cursor:pointer}.controls{position:fixed;left:50%;bottom:max(16px,env(safe-area-inset-bottom));transform:translateX(-50%);display:none;grid-template-columns:repeat(3,54px);gap:7px}.controls button{width:54px;height:48px;border:1px solid #fff4;background:var(--glass);color:white;border-radius:15px;font-size:20px;font-weight:900;touch-action:manipulation}.controls [data-key="ArrowUp"]{grid-column:2}.controls [data-key="ArrowLeft"]{grid-column:1}.controls [data-key="ArrowDown"]{grid-column:2}.controls [data-key="ArrowRight"]{grid-column:3}@media(pointer:coarse),(max-width:700px){.controls{display:grid}.brand{font-size:13px}.score b{font-size:20px}.center-note{top:16%;font-size:11px}}
</style><script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script></head><body>
<div class="hud"><div class="brand">Sunny Crossing<small>Ultra component slice</small></div><div class="score"><b id="score">0000</b><span>BEST <i id="best">0000</i></span></div></div>
<div class="center-note" id="note">Arrow keys / WASD to hop</div>
<div class="overlay" id="overlay"><section><h1>Road<br>Blocked</h1><p id="final-score">Score 0000</p><button id="restart">Hop again</button></section></div>
<div class="controls" aria-label="Touch controls"><button data-key="ArrowUp">↑</button><button data-key="ArrowLeft">←</button><button data-key="ArrowDown">↓</button><button data-key="ArrowRight">→</button></div>
<script>
const SPECS=__SPECS__,COMPONENT_HASHES=__HASHES__;
const roadScope={};new Function('window',__ROAD__)(roadScope);
const vehicleScope={};new Function('window',__VEHICLE__)(vehicleScope);
const characterScope={};new Function('window',__CHARACTER__)(characterScope);
if(typeof roadScope.buildPreview!=='function'||typeof vehicleScope.buildPreview!=='function'||typeof characterScope.buildPreview!=='function')throw new Error('FinalAssembler did not consume every component builder');
const scene=new THREE.Scene(),P=SPECS.presentation,G=SPECS.gameplay;scene.background=new THREE.Color(P.sky);scene.fog=new THREE.Fog(P.sky,P.fog_near,P.fog_far);
const camera=new THREE.PerspectiveCamera(42,innerWidth/innerHeight,.1,180),renderer=new THREE.WebGLRenderer({antialias:true,powerPreference:'high-performance'});renderer.setPixelRatio(Math.min(devicePixelRatio,1.8));renderer.setSize(innerWidth,innerHeight);renderer.shadowMap.enabled=true;renderer.shadowMap.type=THREE.PCFSoftShadowMap;renderer.toneMapping=THREE.ACESFilmicToneMapping;renderer.toneMappingExposure=.92;if(THREE.sRGBEncoding)renderer.outputEncoding=THREE.sRGBEncoding;document.body.prepend(renderer.domElement);
scene.add(new THREE.HemisphereLight(0xe9f6ff,0x46633c,P.ambient_intensity||.50));const sun=new THREE.DirectionalLight(0xffe7bd,P.key_intensity||1.34);sun.position.set(-10,18,12);sun.castShadow=true;sun.shadow.mapSize.set(2048,2048);sun.shadow.camera.left=-18;sun.shadow.camera.right=18;sun.shadow.camera.top=24;sun.shadow.camera.bottom=-24;scene.add(sun);const rim=new THREE.DirectionalLight(0xb9d9ff,P.rim_intensity||.24);rim.position.set(12,8,-14);scene.add(rim);
const context={THREE,scene,camera,renderer};const roadRoot=roadScope.buildPreview(context);const vehicleController=vehicleScope.buildPreview(context),characterController=characterScope.buildPreview(context);
if(!vehicleScope.CompiledVehicleAPI||!characterScope.CompiledCharacterAPI)throw new Error('typed component API missing after exact package consumption');
const grassMat=new THREE.MeshStandardMaterial({color:P.grass,roughness:1}),grassGeo=new THREE.BoxGeometry(12,.22,100);for(const x of [-11,11]){const g=new THREE.Mesh(grassGeo,grassMat);g.position.set(x,-.2,0);g.receiveShadow=true;scene.add(g)}
const env=new THREE.Group();env.name='PresentationEnvironment';scene.add(env);const trunkMat=new THREE.MeshStandardMaterial({color:0x76513b,roughness:1}),leafMats=[0x3c7b47,0x5f984e,0x89b957].map(color=>new THREE.MeshStandardMaterial({color,roughness:.9}));
for(let i=0;i<P.environment_density;i++){const side=i%2?-1:1,x=side*(6.8+(i%3)*1.4),z=-44+i*(88/(P.environment_density-1));const trunk=new THREE.Mesh(new THREE.CylinderGeometry(.18,.25,1.8,8),trunkMat);trunk.position.set(x,.75,z);trunk.castShadow=true;env.add(trunk);const crown=new THREE.Mesh(new THREE.DodecahedronGeometry(.82+(i%4)*.08,0),leafMats[i%leafMats.length]);crown.position.set(x,2.0,z);crown.scale.y=1.25;crown.castShadow=true;env.add(crown);if(i%4===0){const rock=new THREE.Mesh(new THREE.DodecahedronGeometry(.38,0),new THREE.MeshStandardMaterial({color:0x89928b,roughness:.95}));rock.position.set(x-side*1.15,.2,z+1.1);rock.scale.set(1.5,.8,1);env.add(rock)}}
const fenceMat=new THREE.MeshStandardMaterial({color:0xb98757,roughness:.96}),fenceSegments=P.fence_segments||8;for(const side of [-1,1])for(let i=0;i<fenceSegments;i++){const z=-34+i*(68/(fenceSegments-1)),x=side*6.35;const post=new THREE.Mesh(new THREE.BoxGeometry(.16,1.0,.16),fenceMat);post.position.set(x,.42,z);post.castShadow=true;env.add(post);if(i<fenceSegments-1)for(const y of [.34,.72]){const rail=new THREE.Mesh(new THREE.BoxGeometry(.12,.10,68/(fenceSegments-1)),fenceMat);rail.position.set(x,y,z+34/(fenceSegments-1));env.add(rail)}}
const flowerColors=[0xffd34e,0xff7a86,0xf3efff],flowerMats=flowerColors.map(color=>new THREE.MeshStandardMaterial({color,roughness:.8}));for(let i=0;i<(P.flower_clusters||8);i++){const side=i%2?-1:1,x=side*(7.1+(i%3)*.42),z=-30+i*7.2;for(let p=0;p<5;p++){const bloom=new THREE.Mesh(new THREE.DodecahedronGeometry(.08,0),flowerMats[(i+p)%3]);bloom.position.set(x+(p-2)*.12,.12,z+Math.sin(p*2.2)*.22);env.add(bloom)}}
const hayMat=new THREE.MeshStandardMaterial({color:0xd7a73d,roughness:1});for(let i=0;i<(P.hay_bales||3);i++){const side=i%2?-1:1,bale=new THREE.Mesh(new THREE.CylinderGeometry(.42,.42,.75,18),hayMat);bale.rotation.z=Math.PI/2;bale.position.set(side*8.2,.42,-18+i*18);bale.castShadow=true;env.add(bale)}
const traffic=new THREE.Group();traffic.name='TrafficSystem';scene.add(traffic);const prototype=vehicleScope.CompiledVehicleAPI.root;if(prototype.parent)prototype.parent.remove(prototype);const cars=[],laneXs=[-3.75,-1.25,1.25,3.75],colors=[0xe84b36,0x3478d4,0xf1b735,0x47a65b,0xb24bc7];
for(let lane=0;lane<4;lane++)for(let j=0;j<3;j++){const car=prototype.clone(true);car.scale.setScalar(.46);car.position.set(laneXs[lane],.08,-25+j*18+(lane%2)*6);car.rotation.y=lane%2?Math.PI:0;car.userData={lane,direction:lane%2?-1:1,speed:G.traffic_speeds[lane],radiusX:.72,radiusZ:1.22};car.traverse(item=>{if(item.isMesh){item.castShadow=true;item.material=item.material.clone();if(['lower-body','belt-body','hood','cabin','rear-deck','fender-shoulder'].includes(item.name))item.material.color.setHex(colors[(lane+j)%colors.length])}});traffic.add(car);cars.push(car)}
function resetTrafficPositions(){cars.forEach((car,index)=>{const lane=car.userData.lane,j=index%3;car.position.set(laneXs[lane],.08,-25+j*18+(lane%2)*6)})}
const player=characterScope.CompiledCharacterAPI.root;player.scale.setScalar(P.character_scale||.53);player.position.set(-4.45,.08,0);const state={mode:'ready',score:0,best:Number(localStorage.getItem('sunny-best')||0),crossings:0,hopping:false,hopStart:0,from:new THREE.Vector3(),to:new THREE.Vector3(),lastTime:performance.now(),started:false,graceUntil:0,trafficPaused:false};
const scoreEl=document.querySelector('#score'),bestEl=document.querySelector('#best'),note=document.querySelector('#note'),overlay=document.querySelector('#overlay'),finalScore=document.querySelector('#final-score');bestEl.textContent=String(state.best).padStart(4,'0');
function updateHud(){scoreEl.textContent=String(state.score).padStart(4,'0');bestEl.textContent=String(state.best).padStart(4,'0')}
function burst(color,count=12){for(let i=0;i<count;i++){const p=new THREE.Mesh(new THREE.IcosahedronGeometry(.06,0),new THREE.MeshBasicMaterial({color,transparent:true}));p.position.copy(player.position).add(new THREE.Vector3((Math.random()-.5)*.5,.3,(Math.random()-.5)*.5));p.userData.v=new THREE.Vector3((Math.random()-.5)*2,1+Math.random()*2,(Math.random()-.5)*2);p.userData.life=.65;scene.add(p);particles.push(p)}}const particles=[];
function move(dx,dz){if(state.mode==='gameover'||state.hopping)return false;if(!state.started){state.started=true;state.mode='playing';state.graceUntil=performance.now()+2500;note.style.opacity=0}state.hopping=true;state.hopStart=performance.now();state.from.copy(player.position);state.to.copy(player.position);state.to.x=THREE.MathUtils.clamp(state.to.x+dx*G.hop_step,-4.55,4.55);state.to.z=THREE.MathUtils.clamp(state.to.z+dz*G.hop_step,-20,20);burst(0xf8dda8,6);return true}
function restart(){state.mode='ready';state.score=0;state.crossings=0;state.hopping=false;state.started=false;state.graceUntil=performance.now()+2500;state.trafficPaused=false;resetTrafficPositions();player.position.set(-4.45,.08,0);characterScope.CompiledCharacterAPI.updateAnimation('idle',0);overlay.classList.remove('show');note.style.opacity=1;updateHud()}
function collide(){if(state.mode==='gameover')return;state.mode='gameover';state.hopping=false;state.best=Math.max(state.best,state.score);localStorage.setItem('sunny-best',state.best);characterScope.CompiledCharacterAPI.updateAnimation('hit',1);burst(0xff694f,28);finalScore.textContent='Score '+String(state.score).padStart(4,'0');overlay.classList.add('show');updateHud()}
function checkCollision(){if(performance.now()<state.graceUntil)return false;for(const car of cars)if(Math.abs(car.position.x-player.position.x)<.68&&Math.abs(car.position.z-player.position.z)<1.18){collide();return true}return false}
function completeCrossing(){if(player.position.x>=4.4){state.crossings++;state.score+=G.score_per_crossing;state.best=Math.max(state.best,state.score);player.position.x=-4.45;burst(P.hud_accent,18);updateHud()}}
function keyName(key){return ({w:'ArrowUp',a:'ArrowLeft',s:'ArrowDown',d:'ArrowRight',W:'ArrowUp',A:'ArrowLeft',S:'ArrowDown',D:'ArrowRight'})[key]||key}function act(key){key=keyName(key);if(key==='ArrowRight')return move(1,0);if(key==='ArrowLeft')return move(-1,0);if(key==='ArrowUp')return move(0,-1);if(key==='ArrowDown')return move(0,1);if((key==='Enter'||key===' ')&&state.mode==='gameover'){restart();return true}return false}
addEventListener('keydown',e=>{if(act(e.key))e.preventDefault()});document.querySelectorAll('.controls button').forEach(button=>button.addEventListener('pointerdown',e=>{e.preventDefault();act(button.dataset.key)}));document.querySelector('#restart').addEventListener('click',restart);
function tick(now){const dt=Math.min(.04,(now-state.lastTime)/1000);state.lastTime=now;if(state.hopping){const duration=G.hop_duration*1000,t=Math.min(1,(now-state.hopStart)/duration),ease=t*t*(3-2*t);player.position.lerpVectors(state.from,state.to,ease);characterScope.CompiledCharacterAPI.updateAnimation('hop',t);if(t>=1){state.hopping=false;characterScope.CompiledCharacterAPI.updateAnimation('idle',0);completeCrossing()}}else if(state.mode!=='gameover')characterScope.CompiledCharacterAPI.updateAnimation('idle',(now%1200)/1200);
if(state.mode!=='gameover'){const factor=1+state.crossings*G.difficulty_gain;if(!state.trafficPaused)for(const car of cars){car.position.z+=car.userData.direction*car.userData.speed*factor*dt;if(car.position.z>29)car.position.z=-29;if(car.position.z<-29)car.position.z=29}checkCollision()}
for(let i=particles.length-1;i>=0;i--){const p=particles[i];p.userData.life-=dt;p.position.addScaledVector(p.userData.v,dt);p.userData.v.y-=4*dt;p.material.opacity=Math.max(0,p.userData.life/.65);if(p.userData.life<=0){scene.remove(p);particles.splice(i,1)}}
const focus=new THREE.Vector3(player.position.x*.25,1.1,player.position.z*.12);camera.position.lerp(new THREE.Vector3(focus.x+P.camera_distance*.62,P.camera_height,focus.z+P.camera_distance),.045);camera.lookAt(focus);renderer.render(scene,camera);requestAnimationFrame(tick)}
window.GameAPI={componentHashes:COMPONENT_HASHES,specs:SPECS,state,snapshot:()=>({mode:state.mode,score:state.score,best:state.best,crossings:state.crossings,player:{x:player.position.x,y:player.position.y,z:player.position.z},cars:cars.length}),move:(direction)=>act(direction),restart,setTrafficPaused:value=>{state.trafficPaused=Boolean(value);if(state.trafficPaused)cars.forEach((car,index)=>{car.position.z=14+(index%3)*5});return state.trafficPaused},forceCollision:()=>{state.graceUntil=0;const car=cars[0];car.position.copy(player.position);checkCollision()},isReady:()=>Boolean(roadRoot&&vehicleScope.CompiledVehicleAPI&&characterScope.CompiledCharacterAPI)};window.__ULTRA_READY__=true;updateHud();requestAnimationFrame(tick);
addEventListener('resize',()=>{camera.aspect=innerWidth/innerHeight;camera.updateProjectionMatrix();renderer.setSize(innerWidth,innerHeight)});
</script></body></html>'''
    hashes = {name: digest(text.encode("utf-8")) for name, text in source.items()}
    html = (template.replace("__ACCENT__", str(specs["presentation"]["hud_accent"]))
            .replace("__SPECS__", json.dumps(specs, separators=(",", ":")))
            .replace("__HASHES__", json.dumps(hashes, separators=(",", ":")))
            .replace("__ROAD__", json.dumps(source["road"]))
            .replace("__VEHICLE__", json.dumps(source["vehicle"]))
            .replace("__CHARACTER__", json.dumps(source["character"])))
    index = output / "index.html"
    index.write_text(html, encoding="utf-8")
    manifest = {
        "schema_name": "PackageConsumptionEvidenceV1",
        "assembler": "deterministic_lane_crossing_slice_v1",
        "output": {"path": str(index), "sha256": digest(index.read_bytes())},
        "consumed": {name: {"path": str(inputs[name]), "sha256": hashes[name]} for name in inputs},
        "specialist_specs": {"path": str(specs_path), "sha256": digest(specs_path.read_bytes()), "values": specs},
    }
    (output / "assembly-manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
