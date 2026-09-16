import json
from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from cinematic_pipeline.config import settings
from cinematic_pipeline.models.schema import StoryboardPlan

DIRECTOR_SYSTEM_INSTRUCTION = """
You are an expert cinematic storyboard director and prompt engineer.
Your task is to take a simple story premise and decompose it into exactly 3 consecutive, visually stunning cinematic scenes.

Rules:
1. Extract a clear Character Anchor: define strict, unchanging visual attributes (hair, face, specific clothing).
2. Establish an Art Style Preset that remains constant throughout the 3 scenes.
3. For each scene, specify unique cinematography (shot type, lens feel), dynamic action, and volumetric lighting.
4. Output MUST conform strictly to the requested StoryboardPlan schema.
"""


class SceneDirector:
    def __init__(self) -> None:
        self.client = genai.Client(api_key=settings.GEMINI_API_KEY)

    @retry(
        stop=stop_after_attempt(5),      
        wait=wait_exponential(multiplier=2, min=4, max=20),
        retry=retry_if_exception_type(Exception), 
        reraise=True    )
    async def generate_storyboard(self, premise: str) -> StoryboardPlan:
        prompt = f"Story Premise: {premise}"

        response = await self.client.aio.models.generate_content(
            model=settings.GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=DIRECTOR_SYSTEM_INSTRUCTION,
                response_mime_type="application/json",
                response_schema=StoryboardPlan,
                temperature=0.7,
            ),
        )

        raw_data = json.loads(response.text)
        storyboard = StoryboardPlan.model_validate(raw_data)
        storyboard.compile_prompts()
        return storyboard
