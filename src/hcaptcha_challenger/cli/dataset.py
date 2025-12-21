import asyncio
import json
import sys
from pathlib import Path

import typer
from playwright.async_api import async_playwright
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
    SpinnerColumn,
)
from rich.table import Table
from typing_extensions import Annotated, TypedDict

from hcaptcha_challenger.agent.collector import CollectorConfig, Collector, check_dataset
from hcaptcha_challenger.models import CaptchaPayload
from hcaptcha_challenger.utils import SiteKey


class TypeStats(TypedDict):
    """Statistics for each request type."""

    total: int
    valid: int
    invalid: int


class DatasetStats(TypedDict):
    """Statistics for the dataset."""

    total: int
    valid: int
    invalid: int
    types: dict[str, TypeStats]


# Create subcommand application
app = typer.Typer(
    name="dataset",
    help="Dataset Management Tools",
    add_completion=False,
    invoke_without_command=True,
)

DEFAULT_SITE_KEY = SiteKey.user_easy
DEFAULT_DATASET_DIR = Path("dataset")


@app.callback()
def dataset_callback(ctx: typer.Context):
    """
    Dataset subcommand callback. Shows help if no command is provided.
    """
    if ctx.invoked_subcommand is None:
        print(ctx.get_help())
        raise typer.Exit()


async def _create_and_monitor_progress(collector: Collector, max_loops: int):
    """Create and monitor a progress bar for the collector"""
    # 首先启动一个异步任务来准备收集器
    collection_task = None

    try:
        # Show preparation progress bar
        with Progress(
            SpinnerColumn(),
            TextColumn("[init_progress.description]{task.description}"),
            transient=True,
        ) as init_progress:
            init_progress.add_task("Preparing...", total=None)
            await asyncio.sleep(2.5)

        typer.echo(
            f"Starting collector - {json.dumps(collector.config.model_dump(mode='json'), indent=2, ensure_ascii=False)}"
        )

        # After preparation is completed, display the main progress bar and start collecting
        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=30),
            "[progress.percentage]{task.percentage:>3.1f}%",
            "•",
            TaskProgressColumn(),
            "•",
            TimeRemainingColumn(),
        ) as progress:
            task_id = progress.add_task("[cyan]Collecting", total=max_loops)

            collection_task = asyncio.create_task(collector.launch(_by_cli=True))

            last_progress = max_loops
            last_request_type = None

            while not collection_task.done():
                current_progress = collector.remaining_progress
                completed = max_loops - current_progress
                current_request_type = collector.current_request_type

                if current_progress != last_progress or current_request_type != last_request_type:
                    desc = "[cyan]Collecting"
                    if current_request_type:
                        desc += f" - type: {current_request_type}"

                    progress.update(task_id, completed=completed, description=desc)

                    last_progress = current_progress
                    last_request_type = current_request_type

                await asyncio.sleep(0.5)

                if current_progress == 0:
                    break

            progress.update(task_id, completed=max_loops, description="[green]Completed")

            try:
                await asyncio.wait_for(collection_task, timeout=5.0)
            except asyncio.TimeoutError:
                if collection_task and not collection_task.done():
                    collection_task.cancel()
    except Exception as e:
        if collection_task and not collection_task.done():
            collection_task.cancel()
        raise e


async def _launch_collector(
    collector_config: CollectorConfig | None = None,
    *,
    headless: bool = False,
    locale: str = "en-US",
    **kwargs,
):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=headless, **kwargs)
        context = await browser.new_context(locale=locale, **kwargs)
        page = await context.new_page()

        collector = Collector(page, collector_config)

        # Create progress bar
        max_loops = collector_config.MAX_LOOP_COUNT if collector_config else 30

        await _create_and_monitor_progress(collector, max_loops)


@app.command(name="collect")
def collect(
    dataset_dir: Annotated[
        Path, typer.Option(help="Dataset local directory", envvar="DATASET_DIR", show_default=True)
    ] = DEFAULT_DATASET_DIR,
    site_key: Annotated[str, typer.Option(help="Site key", envvar="SITE_KEY")] = DEFAULT_SITE_KEY,
    max_loop_count: Annotated[
        int, typer.Option(help="Maximum loop count", envvar="MAX_LOOP_COUNT")
    ] = 15,
    max_running_time: Annotated[
        float, typer.Option(help="Maximum running time (seconds)", envvar="MAX_RUNNING_TIME")
    ] = 300,
    wait_for_timeout_challenge_view: Annotated[
        float,
        typer.Option(
            help="Waiting for the challenge view to render (millisecond)",
            envvar="WAIT_FOR_TIMEOUT_CHALLENGE_VIEW",
        ),
    ] = 2000,
    headless: Annotated[bool, typer.Option(help="Headless mode")] = True,
    locale: Annotated[str, typer.Option(help="Locale setting")] = "en-US",
):
    """Launch hCaptcha challenge data collector"""
    config = CollectorConfig(
        dataset_dir=dataset_dir.resolve(),
        site_key=site_key,
        MAX_LOOP_COUNT=max_loop_count,
        MAX_RUNNING_TIME=max_running_time,
        WAIT_FOR_TIMEOUT_CHALLENGE_VIEW=wait_for_timeout_challenge_view,
    )

    try:
        asyncio.run(_launch_collector(collector_config=config, headless=headless, locale=locale))
    except KeyboardInterrupt:
        typer.echo("Collector stopped")
        sys.exit(0)
    except Exception as e:
        typer.echo(f"Collector error: {e}")
        sys.exit(1)


@app.command(name="label")
def auto_labeling(
    dataset_dir: Annotated[
        Path, typer.Option(help="Dataset local directory", envvar="DATASET_DIR", show_default=True)
    ] = DEFAULT_DATASET_DIR,
):
    """
    Automatically label image datasets using multimodal large language models.

    Args:
        dataset_dir:

    Returns:

    """
    typer.echo("🤯 Not implemented yet.")


@app.command(name="check")
def check(
    dataset_dir: Annotated[
        Path, typer.Option(help="Dataset local directory", envvar="DATASET_DIR", show_default=True)
    ] = DEFAULT_DATASET_DIR,
):
    """
    Check dataset integrity and generate analysis report
    """
    console = Console()
    captcha_files = list(dataset_dir.rglob("*_captcha.json"))

    if not captcha_files:
        console.print(Panel("[bold red]No dataset files found", title="Dataset Check"))
        return

    errors = []
    dataset_stats: DatasetStats = {
        "total": len(captcha_files),
        "valid": 0,
        "invalid": 0,
        "types": {},
    }

    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=30),
        "[progress.percentage]{task.percentage:>3.1f}%",
        "•",
        TaskProgressColumn(),
        "•",
        TimeRemainingColumn(),
    ) as progress:
        task_id = progress.add_task("[cyan]Checking dataset", total=len(captcha_files))

        for i, captcha_json in enumerate(captcha_files):
            try:
                # Load JSON file to get type information for statistics
                cp = CaptchaPayload.model_validate_json(captcha_json.read_bytes())
                request_type = cp.request_type.value if cp.request_type else "unknown"

                # Update type statistics
                if request_type not in dataset_stats["types"]:
                    dataset_stats["types"][request_type] = TypeStats(total=0, valid=0, invalid=0)
                dataset_stats["types"][request_type]["total"] += 1

                # Perform check
                check_dataset(captcha_json)

                # Check passed, update statistics
                dataset_stats["valid"] += 1
                dataset_stats["types"][request_type]["valid"] += 1

            except Exception as e:
                # Check failed, record error
                error_info = {
                    "file": str(captcha_json.resolve()),
                    "error": str(e),
                    "type": request_type if 'request_type' in locals() else "unknown",
                }
                errors.append(error_info)

                # Update statistics
                dataset_stats["invalid"] += 1
                if 'request_type' in locals() and request_type in dataset_stats["types"]:
                    dataset_stats["types"][request_type]["invalid"] += 1

            # Update progress bar
            progress.update(
                task_id,
                completed=i + 1,
                description=f"[cyan]Checking dataset - {i+1}/{len(captcha_files)}",
            )

    # First display error details if any
    if errors:
        error_table = Table(title="Error Details", box=box.ROUNDED, show_lines=True)
        error_table.add_column("No.", style="cyan", no_wrap=True)
        error_table.add_column("File", style="blue")
        error_table.add_column("Type", style="magenta")
        error_table.add_column("Error", style="red")

        for i, error in enumerate(errors[:10]):  # Only display first 10 errors
            error_table.add_row(str(i + 1), error['file'], error['type'], error['error'])

        console.print(error_table)

        if len(errors) > 10:
            console.print(f"[italic yellow]...and {len(errors)-10} more errors[/italic yellow]")

    # Then generate and display statistics report
    stats_table = Table(title="Dataset Statistics", box=box.ROUNDED)
    stats_table.add_column("Metric", style="cyan")
    stats_table.add_column("Value", style="green")
    stats_table.add_column("Percentage", style="yellow")

    stats_table.add_row("Total Files", str(dataset_stats['total']), "")
    stats_table.add_row(
        "Valid Files",
        str(dataset_stats['valid']),
        (
            f"{dataset_stats['valid']/dataset_stats['total']*100:.1f}%"
            if dataset_stats['total'] > 0
            else "0%"
        ),
    )
    stats_table.add_row(
        "Invalid Files",
        str(dataset_stats['invalid']),
        (
            f"{dataset_stats['invalid']/dataset_stats['total']*100:.1f}%"
            if dataset_stats['total'] > 0
            else "0%"
        ),
    )

    console.print(stats_table)

    # Type statistics
    if dataset_stats["types"]:
        type_table = Table(title="Statistics by Type", box=box.ROUNDED)
        type_table.add_column("Type", style="magenta")
        type_table.add_column("Total", style="blue")
        type_table.add_column("Valid", style="green")
        type_table.add_column("Invalid", style="red")
        type_table.add_column("Valid %", style="yellow")

        for type_name, type_stats in dataset_stats["types"].items():
            valid_percent = (
                type_stats["valid"] / type_stats["total"] * 100 if type_stats["total"] > 0 else 0
            )
            type_table.add_row(
                type_name,
                str(type_stats["total"]),
                str(type_stats["valid"]),
                str(type_stats["invalid"]),
                f"{valid_percent:.1f}%",
            )

        console.print(type_table)

    # Show summary in panel
    valid_percent = (
        dataset_stats['valid'] / dataset_stats['total'] * 100 if dataset_stats['total'] > 0 else 0
    )
    summary = f"[bold]Dataset Integrity:[/bold] {valid_percent:.1f}% valid"
    console.print(Panel(summary, title="Summary", border_style="green"))
