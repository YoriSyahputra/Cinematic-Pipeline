import asyncio
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from cinematic_pipeline.core.orchestrator import PipelineOrchestrator

app = typer.Typer(help="Cinematic Scene Pipeline - Multi-scene storyboard generator CLI")
console = Console()


@app.command()
def generate(
    premise: str = typer.Argument(..., help="Premis cerita yang ingin divisualisasikan"),
) -> None:
    """
    Generate storyboard 3 adegan sinematik terstruktur secara konkuren.
    """
    console.print(
        Panel.fit(
            f"[bold cyan]Premis:[/bold cyan] {premise}",
            title="🎬 Cinematic Scene Pipeline",
            border_style="magenta",
        )
    )

    async def _execute() -> dict:
        orchestrator = PipelineOrchestrator()
        return await orchestrator.run_pipeline(premise)

    with console.status("[bold green]Sedang memproses storyboard & adegan secara paralel...", spinner="dots"):
        result = asyncio.run(_execute())

    char_info = (
        f"[bold yellow]Name:[/bold yellow] {result['character_anchor']['name']}\n"
        f"[bold yellow]Features:[/bold yellow] {result['character_anchor']['visual_features']}\n"
        f"[bold yellow]Wardrobe:[/bold yellow] {result['character_anchor']['wardrobe']}"
    )
    console.print(Panel(char_info, title="👤 Anchor Karakter", border_style="yellow"))
    console.print(
        Panel(result["art_style_preset"], title="🎨 Art Style Preset", border_style="blue")
    )

    table = Table(title="🎞️ Hasil Generasi Adegan", show_lines=True)
    table.add_column("Scene", justify="center", style="bold cyan", width=8)
    table.add_column("Compiled Prompt", style="dim", ratio=3)
    table.add_column("Rendered Image URL", style="green", ratio=2)

    for scene in result["rendered_scenes"]:
        table.add_row(
            str(scene["scene_number"]),
            scene["compiled_prompt"],
            scene["image_url"],
        )

    console.print(table)


if __name__ == "__main__":
    app()
