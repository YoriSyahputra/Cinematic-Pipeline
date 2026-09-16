import asyncio
from cinematic_pipeline.core.director import SceneDirector
from cinematic_pipeline.core.generator import ImageGenerator
from cinematic_pipeline.models.schema import StoryboardPlan

class PipelineOrchestrator:
    def __init__(self) -> None:
        self.director = SceneDirector()
        self.image_generator = ImageGenerator()

    async def run_pipeline(self, premise: str) -> dict:
        """
        Menjalankan end-to-end pipeline:
        1. LLM Director mengurai premis menjadi 3 scene terstruktur & konsisten.
        2. Async Generator merender ketiga scene secara paralel (konkuren).
        """
        print(f"Orchestrator Memulai pipeline untuk premis: '{premise}'")
        
        storyboard: StoryboardPlan = await self.director.generate_storyboard(premise)
        print(f" Orchestrator Storyboard berhasil dibuat dengan gaya: {storyboard.art_style_preset}")
        
        print(f" Orchestrator Mengirim 3 scene ke Image Generation Engine secara paralel...")
        tasks = [
            self.image_generator.generate_scene_image(scene) 
            for scene in storyboard.scenes
        ]
        
        image_urls = await asyncio.gather(*tasks)
        
        results = []
        for scene, url in zip(storyboard.scenes, image_urls):
            results.append({
                "scene_number": scene.scene_number,
                "compiled_prompt": scene.final_compiled_prompt,
                "image_url": url
            })
            
        print(f" Orchestrator Seluruh proses pipeline selesai!")
        
        return {
            "art_style_preset": storyboard.art_style_preset,
            "character_anchor": storyboard.character_anchor.model_dump(),
            "rendered_scenes": results
        }
