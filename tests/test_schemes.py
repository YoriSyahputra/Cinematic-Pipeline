import pytest
from pydantic import ValidationError
from cinematic_pipeline.models.schema import CharacterAnchor, ScenePrompt, StoryboardPlan


def test_character_anchor_creation():
    anchor = CharacterAnchor(
        name="Arthur Vance",
        visual_features="Mid-40s male, weary brown eyes",
        wardrobe="Charcoal wool trench coat",
    )
    assert anchor.name == "Arthur Vance"
    assert "trench coat" in anchor.wardrobe


def test_storyboard_plan_compile_prompts():
    anchor = CharacterAnchor(
        name="Arthur Vance",
        visual_features="Mid-40s male",
        wardrobe="Charcoal coat",
    )
    scenes = [
        ScenePrompt(
            scene_number=i,
            camera_shot="Wide Shot",
            action=f"Action sequence {i}",
            lighting_and_atmosphere="Moody haze",
        )
        for i in range(1, 4)
    ]

    plan = StoryboardPlan(
        art_style_preset="35mm anamorphic neo-noir",
        character_anchor=anchor,
        scenes=scenes,
    )

    plan.compile_prompts()

    # Verifikasi injeksi deterministik token karakter dan style ke seluruh scene
    for scene in plan.scenes:
        assert "35mm anamorphic neo-noir" in scene.final_compiled_prompt
        assert "Arthur Vance" in scene.final_compiled_prompt
        assert "Charcoal coat" in scene.final_compiled_prompt


def test_storyboard_plan_strictly_three_scenes():
    anchor = CharacterAnchor(
        name="Test",
        visual_features="Features",
        wardrobe="Wardrobe",
    )
    # Kurang dari 3 scene harus memicu ValidationError
    invalid_scenes = [
        ScenePrompt(
            scene_number=1,
            camera_shot="Shot",
            action="Action",
            lighting_and_atmosphere="Light",
        )
    ]

    with pytest.raises(ValidationError):
        StoryboardPlan(
            art_style_preset="Preset",
            character_anchor=anchor,
            scenes=invalid_scenes,
        )
