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

        challenge_path_str = str(challenge_path)
        if challenge_path_str.endswith(".challenge"):
            search_pattern = "**/*_model_answer.json"
        else:
            search_pattern = "**/.challenge/**/*_model_answer.json"

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

        # Create dashboard layout
        layout = Layout()
        layout.split_column(Layout(name="header"), Layout(name="main"), Layout(name="footer"))

        # Header - Summary information
        start_time = stats.start_time.strftime("%Y-%m-%d %H:%M:%S") if stats.start_time else "N/A"
        end_time = stats.end_time.strftime("%Y-%m-%d %H:%M:%S") if stats.end_time else "N/A"

        summary_table = Table(show_header=False, box=box.SIMPLE)
        summary_table.add_column(style="cyan bold")
        summary_table.add_column(style="white")

        summary_table.add_row("Time Period:", f"{start_time} → {end_time}")
        summary_table.add_row("Total Challenges:", f"{stats.total_challenges}")
        summary_table.add_row("Total API Calls:", f"{stats.total_files}")

        header_panel = Panel(
            summary_table,
            title="[bold blue]Model Usage Cost Analysis[/bold blue]",
            border_style="blue",
            box=ROUNDED,
        )
        layout["header"].update(header_panel)

        # Main content - Create the main stats table
        cost_table = Table(title="Cost Overview", box=ROUNDED, border_style="blue")
        cost_table.add_column("Metric", style="cyan")
        cost_table.add_column("Value", style="green")

        cost_table.add_row("Total Input Tokens", f"{stats.total_input_tokens:,}")
        cost_table.add_row("Total Output Tokens", f"{stats.total_output_tokens:,}")
        cost_table.add_row("Total API Cost", f"${stats.total_cost:.4f}")
        cost_table.add_row("Average Cost per Challenge", f"${stats.average_cost_per_challenge:.6f}")
        cost_table.add_row("Median Cost per Challenge", f"${stats.median_cost_per_challenge:.6f}")

        # Model details table
        model_table = Table(title="Model Usage Breakdown", box=ROUNDED, border_style="cyan")
        model_table.add_column("Model", style="magenta")
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
                f"${input_cost:.4f}",
                f"${output_cost:.4f}",
                f"${total_cost:.4f}",
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
                f"[bold]${stats.total_cost:.4f}[/bold]",
                "[bold]100%[/bold]",
            )

        # Layout for main section with both tables
        from rich.columns import Columns

        main_content = Columns([cost_table, model_table])
        layout["main"].update(main_content)

        # Footer with additional information
        if output_file:
            footer_text = f"[italic]Detailed statistics saved to: {output_file}[/italic]"
        else:
            footer_text = (
                "[italic]Tip: Use --output-file option to save detailed statistics[/italic]"
            )

        footer_panel = Panel(footer_text, border_style="dim", box=box.SIMPLE)
        layout["footer"].update(footer_panel)

        # Render the complete dashboard
        console.print(layout)

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
