from __future__ import annotations

from agent.component_dsl import (
    CharacterDesignSpecV1,
    VehicleDesignSpecV1,
    compile_character_design_spec,
    compile_vehicle_design_spec,
)


def test_vehicle_spec_clamps_weak_model_values() -> None:
    spec = VehicleDesignSpecV1.from_parts(
        {
            "body": {"paint": "#ff6600", "cabin_taper": 5, "stance": -2},
            "wheels": {"radius": 9, "width": 0.01, "spokes": 99},
            "glass": {"opacity": 1.0, "tint": "#c9a37d"},
            "fascia": {"headlight": "#ff8c00"},
        }
    )

    assert spec.paint == 0xFF6600
    assert spec.cabin_taper == 0.86
    assert spec.stance == 0.50
    assert spec.tire_radius == 0.52
    assert spec.tire_width == 0.24
    assert spec.spoke_count == 8
    assert spec.glass_opacity == 0.68
    assert spec.glass == 0x315E72
    assert spec.headlight == 0xFFF0B2


def test_compiled_vehicle_owns_topology_and_typed_api() -> None:
    source = compile_vehicle_design_spec(VehicleDesignSpecV1())

    assert "window.buildPreview" in source
    assert "window.CompiledVehicleAPI" in source
    assert "new THREE.CylinderGeometry" in source
    assert "tapered('cabin'" in source
    assert "pane('windshield'" in source
    assert "wheelCenters" in source
    assert "forward:'+Z'" in source


def test_character_spec_clamps_weak_model_motion_values() -> None:
    spec = CharacterDesignSpecV1.from_parts(
        {
            "form": {"body_scale": 4, "head_scale": 0.1},
            "motion": {"hop_height": 9, "hop_duration": 0.01, "squash": 2},
            "palette": {"body": "#fff1c4", "accent": "bad"},
        }
    )
    assert spec.body == 0xFFF1C4
    assert spec.body_scale == 1.14
    assert spec.head_scale == 0.88
    assert spec.hop_height == 0.88
    assert spec.hop_duration == 0.26
    assert spec.squash == 0.23
    assert spec.accent == 0xE84B36


def test_compiled_character_has_skeleton_motion_and_factory_api() -> None:
    source = compile_character_design_spec(CharacterDesignSpecV1())
    assert "window.CompiledCharacterAPI" in source
    assert "create:createCharacter" in source
    assert "leftWingPivot" in source
    assert "leftLegPivot" in source
    assert "states:['idle','hop','hit']" in source
    assert "updateAnimation" in source
    assert "contact-shadow" in source
