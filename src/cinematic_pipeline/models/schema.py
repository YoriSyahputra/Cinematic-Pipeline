from pydantic import BaseModel, Field


class CharacterAnchor(BaseModel):
    name: str = Field(description="Nama karakter utama atau identifier unik")
    visual_features: str = Field(
        description="Fitur fisik permanen: usia perkiraan, bentuk wajah, warna & gaya rambut, ekspresi khas"
    )
    wardrobe: str = Field(
        description="Pakaian detail, tekstur bahan, aksesori statis yang wajib ada di setiap shot"
    )


class ScenePrompt(BaseModel):
    scene_number: int = Field(description="Urutan adegan (1, 2, atau 3)")
    camera_shot: str = Field(
        description="Teknik sinematografi, misal: 'Extreme Wide Shot', 'Dutch Angle Close-Up', 'Tracking Shot'"
    )
    action: str = Field(
        description="Aktivitas spesifik karakter dan dinamika latar tempat pada shot ini"
    )
    lighting_and_atmosphere: str = Field(
        description="Pencahayaan dan atmosfer, misal: 'Volumetric god rays through haze', 'Neon reflections on wet pavement'"
    )
    final_compiled_prompt: str = Field(
        default="",
        description="Prompt akhir gabungan yang akan langsung diumpankan ke model gambar"
    )


class StoryboardPlan(BaseModel):
    art_style_preset: str = Field(
        description="Token gaya visual global, misal: '35mm anamorphic photography, Kodak Portra 400, grainy texture'"
    )
    character_anchor: CharacterAnchor
    scenes: list[ScenePrompt] = Field(
        min_length=3,
        max_length=3,
        description="Tepat 3 adegan sekuensial yang saling berkesinambungan"
    )

    def compile_prompts(self) -> None:
        """
        Menyatukan anchor visual, style preset, dan aksi tiap scene
        untuk memastikan determinisme konsistensi karakter.
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
