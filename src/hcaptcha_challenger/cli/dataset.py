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

from hcaptcha_challenger.agent.collector import CollectorConfig, Collector, check_dataset
from hcaptcha_challenger.models import CaptchaPayload
from hcaptcha_challenger.utils import SiteKey

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
    ] = DEFAULT_DATASET_DIR
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
    ] = DEFAULT_DATASET_DIR
):
    """
    检查数据集的完整性并生成分析报告
    """
    captcha_files = list(dataset_dir.rglob("*_captcha.json"))

    if not captcha_files:
        typer.echo("没有找到任何数据集文件")
        return

    errors = []
    dataset_stats = {"total": len(captcha_files), "valid": 0, "invalid": 0, "types": {}}

    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=30),
        "[progress.percentage]{task.percentage:>3.1f}%",
        "•",
        TaskProgressColumn(),
        "•",
        TimeRemainingColumn(),
    ) as progress:
        task_id = progress.add_task("[cyan]检查数据集", total=len(captcha_files))

        for i, captcha_json in enumerate(captcha_files):
            try:
                # 加载JSON文件以获取类型信息，用于统计
                cp = CaptchaPayload.model_validate_json(captcha_json.read_bytes())
                request_type = cp.request_type.value if cp.request_type else "unknown"

                # 更新类型统计
                if request_type not in dataset_stats["types"]:
                    dataset_stats["types"][request_type] = {"total": 0, "valid": 0, "invalid": 0}
                dataset_stats["types"][request_type]["total"] += 1

                # 执行检查
                check_dataset(captcha_json)

                # 检查通过，更新统计
                dataset_stats["valid"] += 1
                dataset_stats["types"][request_type]["valid"] += 1

            except Exception as e:
                # 检查失败，记录错误
                error_info = {
                    "file": str(captcha_json.resolve()),
                    "error": str(e),
                    "type": request_type if 'request_type' in locals() else "unknown",
                }
                errors.append(error_info)

                # 更新统计
                dataset_stats["invalid"] += 1
                if 'request_type' in locals() and request_type in dataset_stats["types"]:
                    dataset_stats["types"][request_type]["invalid"] += 1

            # 更新进度条
            progress.update(
                task_id,
                completed=i + 1,
                description=f"[cyan]检查数据集 - {i+1}/{len(captcha_files)}",
            )

    # 生成报告
    typer.echo("\n数据集检查报告:")
    typer.echo(f"总文件数: {dataset_stats['total']}")
    typer.echo(
        f"有效文件: {dataset_stats['valid']} ({dataset_stats['valid']/dataset_stats['total']*100:.1f}%)"
    )
    typer.echo(
        f"无效文件: {dataset_stats['invalid']} ({dataset_stats['invalid']/dataset_stats['total']*100:.1f}%)"
    )

    if dataset_stats["types"]:
        typer.echo("\n按类型统计:")
        for type_name, type_stats in dataset_stats["types"].items():
            valid_percent = (
                type_stats["valid"] / type_stats["total"] * 100 if type_stats["total"] > 0 else 0
            )
            typer.echo(
                f"  - {type_name}: 共{type_stats['total']}个，有效{type_stats['valid']}个 ({valid_percent:.1f}%)"
            )

    if errors:
        typer.echo("\n错误详情:")
        for i, error in enumerate(errors[:10]):  # 只显示前10个错误
            typer.echo(f"  {i+1}. 文件: {error['file']}")
            typer.echo(f"     类型: {error['type']}")
            typer.echo(f"     错误: {error['error']}")

        if len(errors) > 10:
            typer.echo(f"  ...以及 {len(errors)-10} 个更多错误")
