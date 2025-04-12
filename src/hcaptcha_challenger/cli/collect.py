import asyncio
import json
import sys
from pathlib import Path

import typer
from playwright.async_api import async_playwright
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from typing_extensions import Annotated

from hcaptcha_challenger.agent.collector import CollectorConfig, Collector
from hcaptcha_challenger.utils import SiteKey

# Create subcommand application
app = typer.Typer()

DEFAULT_SITE_KEY = SiteKey.user_easy


async def create_and_monitor_progress(collector, max_loops):
    """Create and monitor a progress bar for the collector"""
    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=50),
        "[progress.percentage]{task.percentage:>3.1f}%",
        "•",
        TaskProgressColumn(),
        "•",
        TimeRemainingColumn(),
    ) as progress:
        task_id = progress.add_task("[cyan]Collecting", total=max_loops)

        # Start a background task to update progress bar
        async def update_progress():
            last_progress = 0
            completed = 0

            while completed < max_loops:
                # Current completed count = total - remaining
                completed = max_loops - collector.remaining_progress

                if completed != last_progress:
                    progress.update(task_id, completed=completed, description=f"[cyan]Collecting")
                    last_progress = completed

                # Short sleep to avoid high CPU usage
                await asyncio.sleep(1)

                # Check if collector has completed
                if collector.remaining_progress == 0:
                    break

        # Create and start progress update task
        progress_task = asyncio.create_task(update_progress())

        # Start collector
        await collector.launch(_by_cli=True)

        # Ensure progress updates to final state
        progress.update(task_id, completed=max_loops, description="[green]Collection completed")

        # Wait for progress update task to complete
        try:
            await asyncio.wait_for(progress_task, timeout=2.0)
        except asyncio.TimeoutError:
            pass  # Task may have completed or been cancelled


async def launch_collector(
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

        await create_and_monitor_progress(collector, max_loops)


@app.command()
def collect(
    dataset_dir: Annotated[
        Path, typer.Option(help="Dataset save directory", envvar="DATASET_DIR", show_default=True)
    ] = Path("dataset"),
    site_key: Annotated[str, typer.Option(help="Site key", envvar="SITE_KEY")] = DEFAULT_SITE_KEY,
    max_loop_count: Annotated[
        int, typer.Option(help="Maximum loop count", envvar="MAX_LOOP_COUNT")
    ] = 5,
    max_running_time: Annotated[
        float, typer.Option(help="Maximum running time (seconds)", envvar="MAX_RUNNING_TIME")
    ] = 300,
    headless: Annotated[bool, typer.Option(help="Headless mode", envvar="HEADLESS")] = True,
    locale: Annotated[str, typer.Option(help="Locale setting")] = "en-US",
):
    """Launch hCaptcha challenge data collector"""
    # Convert types
    config = CollectorConfig(
        dataset_dir=dataset_dir,
        site_key=site_key,
        MAX_LOOP_COUNT=max_loop_count,
        MAX_RUNNING_TIME=max_running_time,
    )

    typer.echo(
        f"Starting collector - Config: {json.dumps(config.model_dump(mode='json'), indent=2, ensure_ascii=False)}"
    )

    # Launch collector
    try:
        asyncio.run(launch_collector(collector_config=config, headless=headless, locale=locale))
    except KeyboardInterrupt:
        typer.echo("Collector stopped")
        sys.exit(0)
    except Exception as e:
        typer.echo(f"Collector error: {e}")
        sys.exit(1)
