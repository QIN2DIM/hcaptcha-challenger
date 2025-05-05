from pathlib import Path
from typing import Annotated, Optional

import typer
from rich import box
from rich.box import ROUNDED
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table

from hcaptcha_challenger.helper.cost_calculator import export_stats

app = typer.Typer()

DEFAULT_CHALLENGE_DIR = Path("tmp")


@app.callback()
def dataset_callback(ctx: typer.Context):
    """
    Dataset subcommand callback. Shows help if no command is provided.
    """
    if ctx.invoked_subcommand is None:
        print(ctx.get_help())
        raise typer.Exit()


@app.command(name="cost")
def check_cost(
    challenge_dir: Annotated[
        Path,
        typer.Option(
            help="Challenge directory to analyze", envvar="CHALLENGE_DIR", show_default=True
        ),
    ] = DEFAULT_CHALLENGE_DIR,
    output_file: Annotated[
        Optional[Path], typer.Option(help="Save stats to JSON file (optional)")
    ] = None,
    show_all_models: Annotated[
        bool, typer.Option("--all", "-a", help="Show details for all models, even with low usage")
    ] = False,
    threshold: Annotated[
        int,
        typer.Option(help="Minimum usage count to show detailed model stats", show_default=True),
    ] = 5,
):
    """
    Calculate and display model usage costs for challenges
    """
    console = Console()

    try:
        # Check if directory exists
        challenge_path = Path(challenge_dir).resolve()
        if not challenge_path.exists():
            console.print(
                Panel(
                    f"[bold red]Directory not found: {challenge_path}",
                    title="Error",
                    border_style="red",
                )
            )
            raise typer.Exit(1)

        # Use a more comprehensive search pattern to find all model answer files
        # regardless of nested structure
        search_pattern = "**/*_model_answer.json"

        # Count model answer files to check if any exist
        answer_files = list(challenge_path.glob(search_pattern))
        if not answer_files:
            console.print(
                Panel(
                    f"[bold yellow]No model answer files found in {challenge_path}[/bold yellow]\n"
                    f"Make sure the directory contains challenge data with *_model_answer.json files.",
                    title="No Data Found",
                    border_style="yellow",
                )
            )
            raise typer.Exit(1)

        with console.status(f"[bold blue]Analyzing {len(answer_files)} model answer files..."):
            # Calculate model usage statistics
            stats = export_stats(challenge_path, output_file)

        # Create a compact, integrated summary table
        summary_table = Table(
            title="[bold blue]Model Usage Cost Analysis[/bold blue]",
            box=box.ROUNDED,
            border_style="blue",
            padding=(0, 1),
            width=None,
        )

        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="green")

        # Add summary information
        summary_table.add_row("Total Challenges", f"{stats.total_challenges:,}")
        summary_table.add_row("Total API Calls", f"{stats.total_files:,}")
        summary_table.add_row("Total Input Tokens", f"{stats.total_input_tokens:,}")
        summary_table.add_row("Total Output Tokens", f"{stats.total_output_tokens:,}")
        summary_table.add_row("Total API Cost", f"${stats.total_cost:.3f}")
        summary_table.add_row(
            "Average Cost per Challenge", f"${stats.average_cost_per_challenge:.3f}"
        )
        summary_table.add_row(
            "Median Cost per Challenge", f"${stats.median_cost_per_challenge:.3f}"
        )

        # Model details table with more compact design
        model_table = Table(
            title="Model Usage Breakdown",
            box=box.ROUNDED,
            border_style="cyan",
            padding=(0, 1),
            width=None,
        )

        model_table.add_column("Model", style="magenta", no_wrap=True)
        model_table.add_column("Calls", style="yellow", justify="right")
        model_table.add_column("Input Tokens", style="green", justify="right")
        model_table.add_column("Output Tokens", style="green", justify="right")
        model_table.add_column("Input Cost", style="cyan", justify="right")
        model_table.add_column("Output Cost", style="cyan", justify="right")
        model_table.add_column("Total Cost", style="blue bold", justify="right")
        model_table.add_column("% of Total", style="red", justify="right")

        # Filter and sort models by usage
        filtered_models = {}
        for model, data in stats.model_details.items():
            if model == "Total":
                continue

            if show_all_models or data.get("usage_count", 0) >= threshold:
                filtered_models[model] = data

        # Sort by cost (highest first)
        sorted_models = sorted(
            filtered_models.items(), key=lambda x: x[1].get("total_cost", 0), reverse=True
        )

        for model, data in sorted_models:
            usage_count = data.get("usage_count", 0)
            input_tokens = data.get("input_tokens", 0)
            output_tokens = data.get("output_tokens", 0)
            input_cost = data.get("input_cost", 0)
            output_cost = data.get("output_cost", 0)
            total_cost = data.get("total_cost", 0)
            percentage = (total_cost / stats.total_cost * 100) if stats.total_cost > 0 else 0

            model_table.add_row(
                model,
                f"{usage_count:,}",
                f"{input_tokens:,}",
                f"{output_tokens:,}",
                f"${input_cost:.3f}",
                f"${output_cost:.3f}",
                f"${total_cost:.3f}",
                f"{percentage:.1f}%",
            )

        # Add a total row
        if "Total" in stats.model_details:
            total_data = stats.model_details["Total"]
            model_table.add_row(
                "[bold]Total[/bold]",
                f"[bold]{stats.total_files:,}[/bold]",
                f"[bold]{stats.total_input_tokens:,}[/bold]",
                f"[bold]{stats.total_output_tokens:,}[/bold]",
                "",
                "",
                f"[bold]${stats.total_cost:.3f}[/bold]",
                "[bold]100%[/bold]",
            )

        # Print tables directly instead of using Layout
        console.print(summary_table)
        console.print(model_table)

        # Display model threshold notice if needed
        if not show_all_models and len(stats.model_details) - 1 > len(filtered_models):
            hidden_models = len(stats.model_details) - 1 - len(filtered_models)
            console.print(
                f"[dim italic]* {hidden_models} models with usage count below {threshold} are hidden. "
                f"Use --all flag to show all models.[/dim italic]"
            )

    except Exception as e:
        console.print(Panel(f"[bold red]Error: {str(e)}", title="Error", border_style="red"))
        raise typer.Exit(1)
