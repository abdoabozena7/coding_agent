"""Typed visual component specifications compiled into safe Three.js geometry.

Small local models are good at choosing bounded design parameters but brittle at
repeating scene-graph plumbing.  These compilers keep aesthetic decisions in
the specialist specs while the harness owns topology, coordinate conventions,
and typed integration APIs.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any, Mapping


def _clamp(value: Any, low: float, high: float, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return max(low, min(high, number))


def _color(value: Any, default: int) -> int:
    if isinstance(value, int):
        return max(0, min(0xFFFFFF, value))
    text = str(value or "").strip().lower().replace("#", "").replace("0x", "")
    if re.fullmatch(r"[0-9a-f]{6}", text):
        return int(text, 16)
    return default


def _cool_glass(value: Any, default: int) -> int:
    color = _color(value, default)
    red, green, blue = (color >> 16) & 255, (color >> 8) & 255, color & 255
    return color if blue >= red * 0.86 and blue >= green * 0.82 else default


def _bright_headlight(value: Any, default: int) -> int:
    color = _color(value, default)
    red, green, blue = (color >> 16) & 255, (color >> 8) & 255, color & 255
    return color if red >= 220 and green >= 180 and blue >= 100 else default


def _shade(color: int, factor: float) -> int:
    red, green, blue = (color >> 16) & 255, (color >> 8) & 255, color & 255
    return (
        int(red * factor) << 16
        | int(green * factor) << 8
        | int(blue * factor)
    )


@dataclass(frozen=True, slots=True)
class VehicleDesignSpecV1:
    paint: int = 0xE34B3F
    paint_secondary: int = 0xA9272D
    trim: int = 0x202A32
    glass: int = 0x315E72
    rim: int = 0xD6DCE2
    headlight: int = 0xFFF0B2
    taillight: int = 0xE31B36
    tire_radius: float = 0.46
    tire_width: float = 0.30
    spoke_count: int = 6
    cabin_taper: float = 0.76
    hood_slope: float = 0.82
    glass_opacity: float = 0.58
    stance: float = 0.56
    style_name: str = "sunlit rally hatch"

    @classmethod
    def from_parts(cls, parts: Mapping[str, Mapping[str, Any]]) -> "VehicleDesignSpecV1":
        defaults = cls()
        body = dict(parts.get("body") or {})
        wheels = dict(parts.get("wheels") or {})
        glass = dict(parts.get("glass") or {})
        fascia = dict(parts.get("fascia") or {})
        return cls(
            paint=_color(body.get("paint"), defaults.paint),
            paint_secondary=_color(body.get("paint_secondary"), defaults.paint_secondary),
            trim=_color(fascia.get("trim", body.get("trim")), defaults.trim),
            glass=_cool_glass(glass.get("tint"), defaults.glass),
            rim=_color(wheels.get("rim"), defaults.rim),
            headlight=_bright_headlight(fascia.get("headlight"), defaults.headlight),
            taillight=_color(fascia.get("taillight"), defaults.taillight),
            tire_radius=_clamp(wheels.get("radius"), 0.40, 0.52, defaults.tire_radius),
            tire_width=_clamp(wheels.get("width"), 0.24, 0.34, defaults.tire_width),
            spoke_count=int(_clamp(wheels.get("spokes"), 5, 8, defaults.spoke_count)),
            cabin_taper=_clamp(body.get("cabin_taper"), 0.68, 0.86, defaults.cabin_taper),
            hood_slope=_clamp(body.get("hood_slope"), 0.72, 0.92, defaults.hood_slope),
            glass_opacity=_clamp(glass.get("opacity"), 0.42, 0.68, defaults.glass_opacity),
            stance=_clamp(body.get("stance"), 0.50, 0.64, defaults.stance),
            style_name=str(body.get("style_name") or defaults.style_name)[:80],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_name": "VehicleDesignSpecV1",
            "paint": f"#{self.paint:06x}",
            "paint_secondary": f"#{self.paint_secondary:06x}",
            "trim": f"#{self.trim:06x}",
            "glass": f"#{self.glass:06x}",
            "rim": f"#{self.rim:06x}",
            "headlight": f"#{self.headlight:06x}",
            "taillight": f"#{self.taillight:06x}",
            "tire_radius": self.tire_radius,
            "tire_width": self.tire_width,
            "spoke_count": self.spoke_count,
            "cabin_taper": self.cabin_taper,
            "hood_slope": self.hood_slope,
            "glass_opacity": self.glass_opacity,
            "stance": self.stance,
            "style_name": self.style_name,
        }


def compile_vehicle_design_spec(spec: VehicleDesignSpecV1) -> str:
    """Compile one cohesive, typed vehicle component from specialist choices."""

    values = {
        "paint": spec.paint,
        "paint2": spec.paint_secondary,
        "accent": _shade(spec.paint_secondary, 0.68),
        "shadow": _shade(spec.paint, 0.62),
        "trim": spec.trim,
        "glass": spec.glass,
        "rim": spec.rim,
        "head": spec.headlight,
        "tail": spec.taillight,
        "radius": round(spec.tire_radius, 4),
        "width": round(spec.tire_width, 4),
        "spokes": spec.spoke_count,
        "taper": round(spec.cabin_taper, 4),
        "hood": round(spec.hood_slope, 4),
        "opacity": round(spec.glass_opacity, 4),
        "stance": round(spec.stance, 4),
        "style": spec.style_name,
    }
    data = json.dumps(values, separators=(",", ":"))
    return f"""window.buildPreview=({{THREE,scene}})=>{{
const S={data},root=new THREE.Group();root.name='CompiledVehicle__'+S.style;
const mat=(color,rough=.55,metal=.05,extra={{}})=>new THREE.MeshStandardMaterial({{color,roughness:rough,metalness:metal,...extra}});
const M={{paint:mat(S.paint,.32,.18),paint2:mat(S.accent,.42,.12),shadow:mat(S.shadow,.5,.14),trim:mat(S.trim,.72,.28),rubber:mat(0x15191d,.9,.02),rim:mat(S.rim,.24,.72),glass:mat(S.glass,.12,.12,{{transparent:true,opacity:Math.max(.58,S.opacity),side:THREE.DoubleSide}}),head:mat(S.head,.16,.08,{{emissive:S.head,emissiveIntensity:.85}}),tail:mat(S.tail,.28,.08,{{emissive:S.tail,emissiveIntensity:.65}})}};
const tapered=(name,size,pos,material,topX=1,topZ=1)=>{{const g=new THREE.BoxGeometry(...size,2,2,4),a=g.attributes.position;for(let i=0;i<a.count;i++)if(a.getY(i)>0){{a.setX(i,a.getX(i)*topX);a.setZ(i,a.getZ(i)*topZ);}}g.computeVertexNormals();const m=new THREE.Mesh(g,material);m.name=name;m.position.set(...pos);m.castShadow=m.receiveShadow=true;root.add(m);return m;}};
const box=(name,size,pos,material,rot=[0,0,0])=>{{const m=new THREE.Mesh(new THREE.BoxGeometry(...size),material);m.name=name;m.position.set(...pos);m.rotation.set(...rot);m.castShadow=m.receiveShadow=true;root.add(m);return m;}};
tapered('lower-body',[2.86,.48,4.72],[0,S.stance,0],M.shadow,.9,.96);
tapered('belt-body',[2.72,.52,4.28],[0,S.stance+.42,-.02],M.paint,.86,.92);
tapered('hood',[2.5,.38,1.48],[0,S.stance+.78,1.53],M.paint,S.hood,.82);
tapered('cabin',[2.18,.92,2.02],[0,S.stance+1.16,-.28],M.paint,S.taper,S.taper);
tapered('rear-deck',[2.42,.32,1.08],[0,S.stance+.8,-1.82],M.paint,.88,.86);
for(const x of [-1.33,1.33]){{box('side-skirt',[.16,.2,3.5],[x,S.stance-.18,-.05],M.trim);for(const z of [-1.55,1.55])tapered('fender-shoulder',[.28,.34,1.02],[x,S.stance+.15,z],M.paint,.88,.82);}}
for(const x of [-1.415,1.415])box('accent-stripe',[.045,.105,2.55],[x,S.stance+.46,-.12],M.paint2);
for(const x of [-1.421,1.421]){{box('door-inset',[.035,.3,1.22],[x,S.stance+.7,-.28],M.shadow);box('door-handle',[.055,.07,.3],[x*1.01,S.stance+.92,-.48],M.rim);}}
const wheels=new THREE.Group();wheels.name='Wheels';root.add(wheels);
const centers=[[-1.58,S.stance,1.55],[1.58,S.stance,1.55],[-1.58,S.stance,-1.55],[1.58,S.stance,-1.55]];
for(const [index,[x,y,z]] of centers.entries()){{const w=new THREE.Group();w.name='wheel-'+index;w.position.set(x,y,z);wheels.add(w);const tire=new THREE.Mesh(new THREE.CylinderGeometry(S.radius,S.radius,S.width,28),M.rubber);tire.rotation.z=Math.PI/2;tire.castShadow=true;w.add(tire);const rim=new THREE.Mesh(new THREE.CylinderGeometry(S.radius*.62,S.radius*.62,S.width*1.04,24),M.rim);rim.rotation.z=Math.PI/2;w.add(rim);const hub=new THREE.Mesh(new THREE.CylinderGeometry(S.radius*.17,S.radius*.17,S.width*1.12,18),M.trim);hub.rotation.z=Math.PI/2;w.add(hub);for(let i=0;i<S.spokes;i++){{const spoke=new THREE.Mesh(new THREE.BoxGeometry(S.width*1.14,S.radius*.72,.065),M.rim);spoke.position.x=(x<0?-1:1)*S.width*.54;spoke.rotation.x=i*Math.PI/S.spokes;w.add(spoke);}}}}
const glass=new THREE.Group();glass.name='Glass';root.add(glass);const pane=(name,size,pos,rot=[0,0,0])=>{{const m=new THREE.Mesh(new THREE.BoxGeometry(...size),M.glass);m.name=name;m.position.set(...pos);m.rotation.set(...rot);glass.add(m);return m;}};
pane('windshield',[1.74,.62,.055],[0,S.stance+1.35,.78],[-.24,0,0]);pane('rear-window',[1.66,.56,.055],[0,S.stance+1.32,-1.3],[.2,0,0]);for(const x of [-1.02,1.02]){{pane('side-window',[.055,.58,.72],[x,S.stance+1.34,-.24]);box('mirror-shell',[.25,.18,.34],[x*1.34,S.stance+1.18,.52],M.trim);}}for(const [x,z] of [[-1.04,.38],[1.04,.38],[-1.04,-.86],[1.04,-.86]])box('pillar',[.08,.72,.09],[x,S.stance+1.36,z],M.trim);
box('front-bumper',[2.5,.16,.18],[0,S.stance-.01,2.43],M.trim);box('rear-bumper',[2.48,.16,.18],[0,S.stance-.01,-2.43],M.trim);
for(const x of [-.82,.82]){{box('headlight',[.46,.24,.09],[x,S.stance+.58,2.39],M.head);box('taillight',[.4,.22,.09],[x,S.stance+.56,-2.39],M.tail);}}
for(let i=-2;i<=2;i++)box('grille-bar',[.22,.07,.055],[i*.27,S.stance+.35,2.47],M.trim);
box('front-intake',[1.48,.13,.06],[0,S.stance+.16,2.48],M.trim);box('plate',[.48,.15,.035],[0,S.stance+.02,2.525],M.rim);box('roof-trim',[1.62,.07,1.2],[0,S.stance+1.68,-.28],M.trim);
scene.add(root);window.CompiledVehicleAPI={{root,forward:'+Z',wheelCenters:centers.map(([x,y,z])=>({{x,y,z}})),style:S.style,spec:S}};return window.CompiledVehicleAPI;
}};"""


@dataclass(frozen=True, slots=True)
class CharacterDesignSpecV1:
    """Bounded art and motion choices for the lane-crossing protagonist."""

    body: int = 0xFFF4D6
    wing: int = 0xF2D7A0
    accent: int = 0xE84B36
    beak: int = 0xF5A623
    leg: int = 0xD98220
    eye: int = 0x17232B
    body_scale: float = 1.0
    head_scale: float = 1.0
    wing_scale: float = 1.0
    hop_height: float = 0.68
    hop_duration: float = 0.34
    flap_angle: float = 0.72
    squash: float = 0.16
    style_name: str = "sunny courier chicken"

    @classmethod
    def from_parts(cls, parts: Mapping[str, Mapping[str, Any]]) -> "CharacterDesignSpecV1":
        defaults = cls()
        form = dict(parts.get("form") or {})
        palette = dict(parts.get("palette") or {})
        motion = dict(parts.get("motion") or {})
        return cls(
            body=_color(palette.get("body"), defaults.body),
            wing=_color(palette.get("wing"), defaults.wing),
            accent=_color(palette.get("accent"), defaults.accent),
            beak=_color(palette.get("beak"), defaults.beak),
            leg=_color(palette.get("leg"), defaults.leg),
            eye=_color(palette.get("eye"), defaults.eye),
            body_scale=_clamp(form.get("body_scale"), 0.88, 1.14, defaults.body_scale),
            head_scale=_clamp(form.get("head_scale"), 0.88, 1.18, defaults.head_scale),
            wing_scale=_clamp(form.get("wing_scale"), 0.82, 1.18, defaults.wing_scale),
            hop_height=_clamp(motion.get("hop_height"), 0.48, 0.88, defaults.hop_height),
            hop_duration=_clamp(motion.get("hop_duration"), 0.26, 0.46, defaults.hop_duration),
            flap_angle=_clamp(motion.get("flap_angle"), 0.48, 0.92, defaults.flap_angle),
            squash=_clamp(motion.get("squash"), 0.10, 0.23, defaults.squash),
            style_name=str(form.get("style_name") or defaults.style_name)[:80],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_name": "CharacterDesignSpecV1",
            "body": f"#{self.body:06x}", "wing": f"#{self.wing:06x}",
            "accent": f"#{self.accent:06x}", "beak": f"#{self.beak:06x}",
            "leg": f"#{self.leg:06x}", "eye": f"#{self.eye:06x}",
            "body_scale": self.body_scale, "head_scale": self.head_scale,
            "wing_scale": self.wing_scale, "hop_height": self.hop_height,
            "hop_duration": self.hop_duration, "flap_angle": self.flap_angle,
            "squash": self.squash, "style_name": self.style_name,
        }


def compile_character_design_spec(spec: CharacterDesignSpecV1) -> str:
    """Compile one animated chicken with a stable integration contract."""

    data = json.dumps({
        "body": spec.body, "wing": spec.wing, "accent": spec.accent,
        "beak": spec.beak, "leg": spec.leg, "eye": spec.eye,
        "bodyScale": round(spec.body_scale, 4), "headScale": round(spec.head_scale, 4),
        "wingScale": round(spec.wing_scale, 4), "hopHeight": round(spec.hop_height, 4),
        "hopDuration": round(spec.hop_duration, 4), "flapAngle": round(spec.flap_angle, 4),
        "squash": round(spec.squash, 4), "style": spec.style_name,
    }, separators=(",", ":"))
    return f"""window.buildPreview=({{THREE,scene}})=>{{
const S={data};const createCharacter=()=>{{
const root=new THREE.Group();root.name='CompiledCharacter__'+S.style;const bodyPivot=new THREE.Group(),leftWingPivot=new THREE.Group(),rightWingPivot=new THREE.Group(),leftLegPivot=new THREE.Group(),rightLegPivot=new THREE.Group();root.add(bodyPivot);bodyPivot.add(leftWingPivot,rightWingPivot,leftLegPivot,rightLegPivot);
const mat=(color,rough=.72)=>new THREE.MeshStandardMaterial({{color,roughness:rough,metalness:.02}}),M={{body:mat(S.body,.7),wing:mat(S.wing,.78),accent:mat(S.accent,.62),beak:mat(S.beak,.68),leg:mat(S.leg,.72),eye:mat(S.eye,.4),white:mat(0xffffff,.56)}};
const mesh=(parent,name,geometry,material,pos=[0,0,0],scale=[1,1,1],rot=[0,0,0])=>{{const m=new THREE.Mesh(geometry,material);m.name=name;m.position.set(...pos);m.scale.set(...scale);m.rotation.set(...rot);m.castShadow=m.receiveShadow=true;parent.add(m);return m;}},sphere=(parent,name,pos,scale,material,segments=24)=>mesh(parent,name,new THREE.SphereGeometry(1,segments,Math.max(12,segments/2)),material,pos,scale);
sphere(bodyPivot,'torso',[0,1.55,0],[.76*S.bodyScale,.92*S.bodyScale,.68*S.bodyScale],M.body);sphere(bodyPivot,'breast',[0,1.46,.48],[.57,.68,.32],M.white);sphere(bodyPivot,'head',[0,2.46,.18],[.63*S.headScale,.62*S.headScale,.59*S.headScale],M.body);sphere(bodyPivot,'muzzle',[0,2.34,.68],[.39,.28,.25],M.white);mesh(bodyPivot,'beak',new THREE.ConeGeometry(.29,.62,4),M.beak,[0,2.37,1.02],[1,.72,1],[Math.PI/2,0,Math.PI/4]);
for(const x of [-.25,.25]){{sphere(bodyPivot,'eye-white',[x,2.62,.66],[.14,.17,.09],M.white,18);sphere(bodyPivot,'eye-pupil',[x,2.62,.74],[.065,.085,.045],M.eye,16);sphere(bodyPivot,'eye-glint',[x-.018,2.66,.78],[.018,.022,.014],M.white,12);}}for(const [x,y,s] of [[0,3.08,.25],[-.18,2.98,.2],[.18,2.98,.2]])sphere(bodyPivot,'comb',[x,y,.05],[s,.29,s*.8],M.accent,18);sphere(bodyPivot,'wattle',[0,2.08,.65],[.19,.28,.15],M.accent,18);
const wingGeo=new THREE.SphereGeometry(1,22,14);leftWingPivot.position.set(-.66,1.72,.04);rightWingPivot.position.set(.66,1.72,.04);mesh(leftWingPivot,'left-wing',wingGeo,M.wing,[0,-.08,0],[.28*S.wingScale,.64*S.wingScale,.47*S.wingScale],[0,0,.14]);mesh(rightWingPivot,'right-wing',wingGeo,M.wing,[0,-.08,0],[.28*S.wingScale,.64*S.wingScale,.47*S.wingScale],[0,0,-.14]);
for(const [i,[x,y,z,r]] of [[0,[-.34,1.58,-.56,-.18]],[1,[0,1.68,-.68,0]],[2,[.34,1.58,-.56,.18]]])mesh(bodyPivot,'tail-feather-'+i,new THREE.ConeGeometry(.25,.78,5),M.wing,[x,y,z],[1,1,1],[Math.PI*.62,0,r]);
const legGeo=new THREE.CylinderGeometry(.075,.085,.7,12),footGeo=new THREE.CylinderGeometry(.055,.065,.42,10);for(const [pivot,x] of [[leftLegPivot,-.29],[rightLegPivot,.29]]){{pivot.position.set(x,1.0,.02);mesh(pivot,'leg',legGeo,M.leg,[0,-.32,0]);for(const toeX of [-.12,0,.12])mesh(pivot,'toe',footGeo,M.leg,[toeX,-.69,.20],[1,1,1],[Math.PI/2,0,0]);}}
const shadow=sphere(root,'contact-shadow',[0,.04,0],[.66,.035,.48],new THREE.MeshBasicMaterial({{color:0x17232b,transparent:true,opacity:.22}}),20),base={{bodyY:bodyPivot.position.y}};
const updateAnimation=(state='idle',phase=0)=>{{const t=Math.max(0,Math.min(1,Number(phase)||0));bodyPivot.position.y=base.bodyY;bodyPivot.rotation.z=0;bodyPivot.scale.set(1,1,1);leftWingPivot.rotation.z=0;rightWingPivot.rotation.z=0;leftLegPivot.rotation.x=0;rightLegPivot.rotation.x=0;shadow.scale.set(1,1,1);shadow.material.opacity=.22;if(state==='hop'){{const arc=Math.sin(Math.PI*t);bodyPivot.position.y=S.hopHeight*arc;bodyPivot.scale.set(1-S.squash*(1-arc)*.55,1+S.squash*(arc-.25),1-S.squash*(1-arc)*.55);leftWingPivot.rotation.z=S.flapAngle*arc;rightWingPivot.rotation.z=-S.flapAngle*arc;leftLegPivot.rotation.x=rightLegPivot.rotation.x=-.72*arc;shadow.scale.set(1-.34*arc,1,1-.34*arc);shadow.material.opacity=.22-.1*arc;}}else if(state==='hit'){{bodyPivot.rotation.z=-.62*t;leftWingPivot.rotation.z=.92;rightWingPivot.rotation.z=-.92;}}else{{const bob=Math.sin(t*Math.PI*2);bodyPivot.position.y=.035*bob;leftWingPivot.rotation.z=.08*bob;rightWingPivot.rotation.z=-.08*bob;}}}};
const controller={{root,pivots:{{body:bodyPivot,leftWing:leftWingPivot,rightWing:rightWingPivot,leftLeg:leftLegPivot,rightLeg:rightLegPivot}},forward:'+Z',states:['idle','hop','hit'],hopDuration:S.hopDuration,spec:S,updateAnimation}};updateAnimation('hop',.54);return controller;}};
const controller=createCharacter();scene.add(controller.root);window.CompiledCharacterAPI={{...controller,create:createCharacter}};return window.CompiledCharacterAPI;}};"""


__all__ = ["VehicleDesignSpecV1", "compile_vehicle_design_spec", "CharacterDesignSpecV1", "compile_character_design_spec"]
