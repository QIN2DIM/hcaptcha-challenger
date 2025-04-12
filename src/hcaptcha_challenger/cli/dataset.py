import asyncio
import json
import sys
from pathlib import Path

import typer
from playwright.async_api import async_playwright
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
    SpinnerColumn,
)
from typing_extensions import Annotated

from hcaptcha_challenger.agent.collector import CollectorConfig, Collector
from hcaptcha_challenger.utils import SiteKey

# Create subcommand application
app = typer.Typer(
    name="dataset",
    help="Dataset collection tool",
    add_completion=False,
    invoke_without_command=True,
)

DEFAULT_SITE_KEY = SiteKey.user_easy


@app.callback()
def dataset_callback(ctx: typer.Context):
    """
    Dataset subcommand callback. Shows help if no command is provided.
    """
    if ctx.invoked_subcommand is None:
        print(ctx.get_help())
        raise typer.Exit()


async def create_and_monitor_progress(collector, max_loops):
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
                    desc = f"[cyan]Collecting"
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
    ] = 15,
    max_running_time: Annotated[
        float, typer.Option(help="Maximum running time (seconds)", envvar="MAX_RUNNING_TIME")
    ] = 300,
    headless: Annotated[bool, typer.Option(help="Headless mode", envvar="HEADLESS")] = True,
    locale: Annotated[str, typer.Option(help="Locale setting")] = "en-US",
):
    """Launch hCaptcha challenge data collector"""
    # Convert types
    config = CollectorConfig(
        dataset_dir=dataset_dir.resolve(),
        site_key=site_key,
        MAX_LOOP_COUNT=max_loop_count,
        MAX_RUNNING_TIME=max_running_time,
    )

    try:
        asyncio.run(launch_collector(collector_config=config, headless=headless, locale=locale))
    except KeyboardInterrupt:
        typer.echo("Collector stopped")
        sys.exit(0)
    except Exception as e:
        typer.echo(f"Collector error: {e}")
        sys.exit(1)
