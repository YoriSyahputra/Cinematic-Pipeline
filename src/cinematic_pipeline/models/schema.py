from pydantic import BaseModel, Field


class CharacterAnchor(BaseModel):
    name: str = Field(description="Main character name or unique identifier")
    visual_features: str = Field(
        description="Permanent physical features: estimated age, face shape, hair color & style, signature expression"
    )
    wardrobe: str = Field(
        description="Detailed clothing, fabric texture, static accessories required in every shot"
    )


class ScenePrompt(BaseModel):
    scene_number: int = Field(description="Sequence number of the scene (1, 2, or 3)")
    camera_shot: str = Field(
        description="Cinematography technique, e.g., 'Extreme Wide Shot', 'Dutch Angle Close-Up', 'Tracking Shot'"
    )
    action: str = Field(
        description="Specific character activity and background environment dynamics in this shot"
    )
    lighting_and_atmosphere: str = Field(
        description="Lighting and atmosphere, e.g., 'Volumetric god rays through haze', 'Neon reflections on wet pavement'"
    )
    final_compiled_prompt: str = Field(
        default="",
        description="Final combined prompt to be fed directly into the image generation model"
    )


class StoryboardPlan(BaseModel):
    art_style_preset: str = Field(
        description="Global visual style tokens, e.g., '35mm anamorphic photography, Kodak Portra 400, grainy texture'"
    )
    character_anchor: CharacterAnchor
    scenes: list[ScenePrompt] = Field(
        min_length=3,
        max_length=3,
        description="Exactly 3 sequential, mutually consistent scenes"
    )

    def compile_prompts(self) -> None:
        """
        Merge visual anchors, style presets, and actions for each scene
        to ensure deterministic character consistency.
        """
        for scene in self.scenes:
            scene.final_compiled_prompt = (
                f"{self.art_style_preset}, "
                f"Subject: {self.character_anchor.name}, {self.character_anchor.visual_features}, "
                f"wearing {self.character_anchor.wardrobe}. "
                f"Cinematography: {scene.camera_shot}. "
                f"Action: {scene.action}. "
                f"Environment & Light: {scene.lighting_and_atmosphere}"
            )
