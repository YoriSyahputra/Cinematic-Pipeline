import asyncio
import random
from cinematic_pipeline.models.schema import ScenePrompt

class ImageGenerator:
    async def generate_scene_image(self, scene: ScenePrompt) -> str:
        """
        Mensimulasikan async image generation API request untuk sebuah scene.
        Nanti bagian ini bisa diganti dengan endpoint HTTP client (httpx) ke Fal.ai/Replicate.
        """
        render_delay = random.uniform(1.5, 3.0)
        print(f"Rendering Image dengan melakukan Simulasi latensi Jaringan")
        await asyncio.sleep(render_delay)
        print(f"Selesai di render")
        mock_image_url = f"https://api.aicines.mock/render/scene_{scene.scene_number}_{int(asyncio.get_running_loop().time())}.png"
        return mock_image_url
