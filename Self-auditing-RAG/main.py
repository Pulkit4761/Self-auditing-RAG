import sys
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from src.rag.pipeline import RAGPipeline

console = Console()

def cmd_ingest(pipeline: RAGPipeline):
    console.print("\n[bold]Ingesting documents...[/bold]")
    try:
        count = pipeline.ingest()
        console.print(f"[green]Done![/green] Indexed {count} chunks.\n")
    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


def cmd_query(pipeline: RAGPipeline):
    console.print(
        "\n[bold]Self-Auditing RAG[/bold] — type your question "
        "(or 'quit' to exit)\n"
    )

    while True:
        query = console.input("[bold cyan]Query>[/bold cyan] ").strip()
        if not query or query.lower() in ("quit", "exit", "q"):
            console.print("[dim]Goodbye![/dim]")
            break

        with console.status("[bold]Thinking..."):
            result = pipeline.query(query)

        # Decision color
        color_map = {"ACCEPT": "green", "REVISE": "yellow", "REJECT": "red"}
        dec = result.decision.value
        color = color_map.get(dec, "white")

        # Answer panel
        console.print(
            Panel(
                result.final_answer,
                title=f"Answer  [{color}]{dec}[/{color}]",
                border_style=color,
            )
        )

        # Audit table
        table = Table(title="Faithfulness Audit", show_lines=True)
        table.add_column("Sentence", style="white", ratio=3)
        table.add_column("Score", justify="center", width=8)
        table.add_column("Verdict", justify="center", width=12)

        for sv in result.audit.sentence_verdicts:
            v_color = "green" if sv.supported else "red"
            verdict_text = "Supported" if sv.supported else "Unsupported"
            table.add_row(
                sv.sentence,
                f"{sv.score:.3f}",
                f"[{v_color}]{verdict_text}[/{v_color}]",
            )

        console.print(table)
        console.print(
            f"  Overall faithfulness: [bold]{result.audit.faithfulness_score:.3f}[/bold]"
        )
        if result.was_revised:
            console.print(
                "  [yellow]Answer was revised to remove unsupported claims.[/yellow]"
            )
        console.print()


def main():
    if len(sys.argv) < 2:
        console.print(
            "[bold]Usage:[/bold]\n"
            "  python main.py ingest   — index documents from documents/\n"
            "  python main.py query    — ask questions interactively\n"
        )
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == "ingest":
        pipeline = RAGPipeline()
        cmd_ingest(pipeline)
    elif command == "query":
        pipeline = RAGPipeline()
        cmd_query(pipeline)
    else:
        console.print(f"[red]Unknown command:[/red] {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
