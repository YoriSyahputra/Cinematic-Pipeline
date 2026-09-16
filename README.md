# 🎬 Cinematic Scene Pipeline

**Deterministic multi-shot consistency engine for generative video/image pipelines.**

Decomposes a narrative premise into structurally consistent scene prompts by compiling immutable character/style anchors into every generated shot — solving the "subject drift" problem inherent to autoregressive diffusion prompting.

<p align="left">
  <img src="https://img.shields.io/badge/python-3.12%2B-blue?logo=python&logoColor=white" alt="Python 3.12+">
  <img src="https://img.shields.io/badge/package_manager-uv-de4436?logo=astral&logoColor=white" alt="uv">
  <img src="https://img.shields.io/badge/validation-pydantic_v2-e92063?logo=pydantic&logoColor=white" alt="Pydantic v2">
  <img src="https://img.shields.io/badge/CLI-typer-000000?logo=python&logoColor=white" alt="Typer">
  <img src="https://img.shields.io/badge/output-rich-9146ff" alt="Rich">
  <img src="https://img.shields.io/badge/tested_with-pytest-0A9EDC?logo=pytest&logoColor=white" alt="Pytest">
  <img src="https://img.shields.io/badge/license-MIT-lightgrey" alt="License">
</p>

---

## Executive Summary & The Problem

Generative image/video models (SDXL, Flux, Midjourney, Sora-class systems) generate each frame or shot as a **near-independent sampling event**. Even with identical seeds or reference images, small variations in prompt phrasing compound across a sequence, producing what's commonly called **subject drift**: a character's face, wardrobe, or the scene's art direction subtly (or drastically) shifts from shot to shot.

This is not a model bug — it's a consequence of how these systems are typically _prompted_. Most naive pipelines re-describe the subject and style in free text for every scene, relying on an LLM or human to remember and restate details consistently. Language models are non-deterministic text generators; asking them to _re-derive_ consistent details on every call is asking for drift by construction.

**Cinematic Scene Pipeline treats consistency as a data problem, not a prompting problem.**

Instead of asking an LLM to _remember_ what a character looks like, we:

1. Extract a **`CharacterAnchor`** once — a frozen, structured record of facial geometry, wardrobe, and permanent features.
2. Extract a **global `ArtStylePreset`** once — lighting, color grade, lens/rendering style.
3. Let the LLM do what it's actually good at: **narrative decomposition** (turning a premise into N scene beats).
4. **Deterministically inject** the anchor and style tokens into every compiled scene prompt via code — not another LLM call.

The LLM proposes the _story_. Pydantic-validated Python guarantees the _consistency_. Drift is architecturally impossible at the compilation step, because the anchor tokens are string-concatenated from the same frozen object on every scene, not regenerated.

---

## End-to-End System Flow

```
                           ┌────────────────────────┐
   User (CLI)              │   Typer CLI Entrypoint │
   "A retired detective    │   cli.py               │
    solves one last case"  └───────────┬────────────┘
        │                              │
        ▼                              ▼
┌───────────────────┐         ┌──────────────────────────┐
│  Rich Spinner /   │ ◄───────┤  Scene Director          │
│  Progress UI      │         │  core/director.py        │
└───────────────────┘         │  gemini-3.6-flash        │
                              │  + tenacity retry wrapper│
                              └───────────┬──────────────┘
                                          │ raw structured JSON
                                          ▼
                              ┌──────────────────────────┐
                              │  Pydantic V2 Validation  │
                              │  models/schema.py        │
                              │  CharacterAnchor         │
                              │  ScenePrompt (min3/max3) │
                              │  StoryboardPlan          │
                              │  .compile_prompts()      │
                              └───────────┬──────────────┘
                                          │ validated StoryboardPlan
                                          │ (anchors deterministically injected)
                                          ▼
                              ┌──────────────────────────┐
                              │  Concurrent Orchestrator │
                              │  core/orchestrator.py    │
                              │  asyncio.gather(*workers)│
                              └───────────┬──────────────┘
                                          │ N parallel generator.py workers
                                          │ (simulated non-blocking render calls)
                                          ▼
                              ┌──────────────────────────┐
                              │  Rich Terminal UI        │
                              │  Result table + timings  │
                              └──────────────────────────┘
```

The critical architectural decision: **validation happens once, between the LLM and the fan-out**, not per-worker. Every downstream render call consumes an already-consistent, already-typed `ScenePrompt` — workers have zero responsibility for correctness, only for execution.

---

## Engineering Deep Dives

### 1. Deterministic Token Injection vs. Naive Prompt Passing

**Naive approach** (what most tutorial pipelines do):

```python
# Re-describes the character in free text on every call — drift-prone
prompt = llm.generate(f"Describe scene {i} with the detective character")
```

Every call is a fresh sampling event. The word choice, level of detail, and even factual details ("brown coat" vs. "tan trench coat") can vary call to call.

**This project's approach:**

```python
class CharacterAnchor(BaseModel):
    name: str
    facial_geometry: str
    wardrobe: str
    permanent_features: str

class ScenePrompt(BaseModel):
    scene_number: int
    action_description: str  # the only part the LLM is trusted to vary

    def compile(self, anchor: CharacterAnchor, style: ArtStylePreset) -> str:
        # Pure, deterministic string composition — no LLM in this path
        return (
            f"{anchor.facial_geometry}, {anchor.wardrobe}, "
            f"{anchor.permanent_features}, {self.action_description}, "
            f"{style.lighting}, {style.color_grade}, {style.lens}"
        )
```

The LLM is only ever asked to produce the _action beat_ (what happens in this shot). The anchor and style tokens are pulled from a single frozen object and concatenated in code. This is the difference between **asking for consistency** and **guaranteeing it structurally**.

### 2. Concurrency Model & Latency Optimization

Scene generation/rendering calls are I/O-bound (network round-trips to a rendering/inference API). Dispatching them sequentially means total latency scales linearly with scene count:

```
Sequential:  O(N)  → latency = t1 + t2 + t3 + ... + tN
Concurrent:  O(1)  → latency ≈ max(t1, t2, t3, ..., tN)
```

The orchestrator dispatches all validated `ScenePrompt` objects as concurrent coroutines:

```python
async def orchestrate(scenes: list[ScenePrompt]) -> list[RenderResult]:
    tasks = [generate_frame(scene) for scene in scenes]
    return await asyncio.gather(*tasks, return_exceptions=False)
```

For a fixed 3-scene storyboard, this cuts wall-clock time from `t1 + t2 + t3` down to roughly `max(t1, t2, t3)` — a meaningful reduction as scene count or per-call latency grows, and the pattern scales cleanly if the scene bound is ever relaxed beyond 3.

**Trade-off acknowledged:** `asyncio.gather` with default settings fails the whole batch on the first unhandled exception. This is why retry logic lives _inside_ each worker (via `tenacity`) rather than at the orchestrator level — a single transient 503 shouldn't take down two otherwise-successful renders.

### 3. Fault Tolerance & Retry Strategy

Upstream generation APIs are unreliable under load — `503` (capacity) and `429` (rate limit) are expected steady-state noise, not exceptional failures. The director's LLM call is wrapped with `tenacity` exponential backoff:

```python
@retry(
    retry=retry_if_exception_type(TransientAPIError),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(5),
    reraise=True,
)
async def decompose_premise(self, premise: str) -> StoryboardPlan:
    ...
```

Design rationale:

- **Exponential backoff** (not fixed-interval retry) avoids synchronized retry storms against an already-degraded upstream.
- **Bounded attempts** (`stop_after_attempt(5)`) prevent silent infinite hangs — failures surface to the CLI with a clear Rich error panel rather than looping forever.
- **Narrow retry predicate** (`retry_if_exception_type`) ensures we only retry _transient_ failures — a `400` schema validation error from a malformed premise fails fast instead of burning 5 retries on a non-retryable bug.

---

## Project Tree Breakdown

```
cinematic-scene-pipeline/
├── pyproject.toml                  # Project metadata, dependencies, uv-managed lockfile source
├── .env.example                    # Template for GEMINI_API_KEY and runtime config
├── .gitignore
├── src/
│   └── cinematic_pipeline/
│       ├── cli.py                  # Typer entrypoint — argument parsing, Rich spinners/panels/tables
│       ├── config.py               # Pydantic Settings: loads & validates .env into typed config object
│       ├── core/
│       │   ├── director.py         # LLM call (gemini-2.0-flash) + tenacity retry wrapper;
│       │   │                       #   converts a free-text premise into a raw structured response
│       │   ├── generator.py        # Async "render worker" — simulates a non-blocking call to a
│       │   │                       #   downstream image/video generation API
│       │   └── orchestrator.py     # asyncio.gather fan-out/fan-in; owns concurrency + result assembly
│       └── models/
│           └── schema.py           # CharacterAnchor, ArtStylePreset, ScenePrompt, StoryboardPlan
│                                    #   + StoryboardPlan.compile_prompts() (deterministic injection)
└── tests/
    └── test_schemes.py             # Pytest suite: anchor invariants, compile_prompts() correctness,
                                     #   and Pydantic schema bound enforcement (min 3 / max 3 scenes)
```

---

## Installation & Quickstart

This project uses [`uv`](https://github.com/astral-sh/uv) for dependency resolution and virtual environment management — no manual `pip`/`venv` bookkeeping required.

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/cinematic-scene-pipeline.git
cd cinematic-scene-pipeline

# 2. Install uv (skip if already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. Sync the environment — creates .venv and installs pinned dependencies
uv sync

# 4. Configure environment variables
cp .env.example .env
# then edit .env and set:
#   GEMINI_API_KEY=your_google_genai_api_key

# 5. Run the pipeline
uv run python -m cinematic_pipeline.cli "A retired detective takes one last case in a neon-lit city"
```

---

## CLI Usage & Example Terminal Output

**Basic usage:**

```bash
uv run python -m cinematic_pipeline.cli "<your narrative premise>"
```

**Optional flags:**

```bash
uv run python -m cinematic_pipeline.cli \
  "A lone astronaut discovers an abandoned station" \
  --style cinematic-noir \
  --verbose
```

**Example output:**

```
╭──────────────────────── Cinematic Scene Pipeline ───────────────────────╮
│  Premise: "A retired detective takes one last case in a neon-lit city"  │
╰─────────────────────────────────────────────────────────────────────────╯

⠋ Decomposing premise via Scene Director (gemini-2.0-flash)...
✓ Storyboard validated — 3 scenes, 1 character anchor, 1 style preset

⠙ Dispatching 3 scenes concurrently (asyncio.gather)...
✓ All scenes rendered in 2.14s (sequential estimate: 6.02s)

                       Storyboard Render Results
┏━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━┓
┃ Scene ┃ Action Description                ┃ Status  ┃ Latency   ┃
┡━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━┓
│   1   │ Detective enters a rain-slicked    │ ✓ done  │  1.42s    │
│       │ alley beneath flickering neon signs│         │           │
│   2   │ A tense standoff in a dim noodle   │ ✓ done  │  2.14s    │
│       │ bar, steam rising between them     │         │           │
│   3   │ Detective walks away as dawn breaks│ ✓ done  │  1.87s    │
│       │ over the city skyline              │         │           │
└───────┴────────────────────────────────────┴─────────┴───────────┘
  
Compiled prompt (Scene 1):
"weathered angular jawline, deep-set grey eyes | tan trench coat,
scuffed leather gloves | scar above left brow | detective enters a
rain-slicked alley beneath flickering neon signs | low-key noir
lighting | desaturated teal-and-amber grade | 35mm anamorphic lens"
```

---

## Test Suite

Run the full suite with verbose output:

```bash
uv run pytest -v
```

**What's covered** (`tests/test_schemes.py`):

| Test                                           | Verifies                                                                                                                                                                            |
| ---------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `test_character_anchor_required_fields`        | `CharacterAnchor` rejects instantiation when facial geometry, wardrobe, or permanent features are missing — no partial/ambiguous anchors can enter the pipeline.                    |
| `test_scene_prompt_bounds`                     | `StoryboardPlan` enforces the **min 3 / max 3** scene constraint at the schema level — an LLM response with 2 or 5 scenes fails validation before it ever reaches the orchestrator. |
| `test_compile_prompts_determinism`             | Calling `compile_prompts()` twice on the same `StoryboardPlan` produces byte-identical output — proves injection is a pure function of state, not a fresh generative call.          |
| `test_compile_prompts_injects_anchor_tokens`   | Every compiled scene prompt string contains the exact anchor and style tokens — confirms no scene can "lose" a consistency detail.                                                  |
| `test_invalid_premise_raises_validation_error` | Malformed/incomplete LLM responses raise a `pydantic.ValidationError` rather than silently passing through with default/null values.                                                |

The test suite intentionally targets the **schema and compilation layer** rather than mocking live API calls — this is where consistency guarantees are made or broken, and where regressions would silently reintroduce drift.

---

## License

MIT — see [`LICENSE`](./LICENSE) for details.
