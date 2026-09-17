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
        Executing the end-to-end pipeline:
        1. The LLM Director breaks down the premise into three structured and consistent scenes.
        2. The Async Generator renders the three scenes in parallel (concurrently).
        """
        print(f" Orchestrator starts the pipeline for the premises:'{premise}'")
        
        storyboard: StoryboardPlan = await self.director.generate_storyboard(premise)
        print(f" The Orchestrator Storyboard was successfully created with the style:{storyboard.art_style_preset}")
        
        print(f"The Orchestrator sends three scenes to the Image Generation Engine in parallel...")
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
            
        print(f" Orchestrator: The entire pipeline process is complete!")
        
        return {
            "art_style_preset": storyboard.art_style_preset,
            "character_anchor": storyboard.character_anchor.model_dump(),
            "rendered_scenes": results
        }
