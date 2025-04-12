import asyncio
import sys
from pathlib import Path

import typer
from playwright.async_api import async_playwright
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from typing_extensions import Annotated

from hcaptcha_challenger.agent.collector import CollectorConfig, Collector
from hcaptcha_challenger.utils import SiteKey

# Create subcommand application
app = typer.Typer(name="collect", help="Dataset collection tool")

DEFAULT_SITE_KEY = SiteKey.user_easy


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

        # 创建进度条
        max_loops = collector_config.MAX_LOOP_COUNT if collector_config else 30

        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=None),
            "[progress.percentage]{task.percentage:>3.1f}%",
            "•",
            TaskProgressColumn(),
            "•",
            TimeRemainingColumn(),
        ) as progress:
            task_id = progress.add_task("[cyan]采集验证码数据...", total=max_loops)

            # 启动一个后台任务来更新进度条
            async def update_progress():
                last_progress = 0
                completed = 0

                while completed < max_loops:
                    # 当前完成数量 = 总数 - 剩余数量
                    completed = max_loops - collector.remaining_progress

                    if completed != last_progress:
                        # 更新描述以显示当前类型信息
                        request_types = (
                            collector_config.focus_types
                            if collector_config and collector_config.focus_types
                            else ["未知"]
                        )
                        type_info = (
                            ", ".join([t.value for t in request_types])
                            if hasattr(request_types[0], "value")
                            else ", ".join(request_types)
                        )
                        progress.update(
                            task_id,
                            completed=completed,
                            description=f"[cyan]采集验证码数据... [magenta](类型: {type_info})",
                        )
                        last_progress = completed

                    # 短暂休眠避免CPU占用过高
                    await asyncio.sleep(0.3)

                    # 检查收集器是否已完成
                    if collector.remaining_progress == 0:
                        break

            # 创建并启动进度更新任务
            progress_task = asyncio.create_task(update_progress())

            # 启动收集器
            await collector.launch(_by_cli=True)

            # 确保进度更新到最终状态
            progress.update(task_id, completed=max_loops, description="[green]采集完成!")

            # 等待进度更新任务完成
            try:
                await asyncio.wait_for(progress_task, timeout=2.0)
            except asyncio.TimeoutError:
                pass  # 任务可能已经完成或被取消


@app.command()
def collect(
    dataset_dir: Annotated[
        Path, typer.Option(help="Dataset save directory", envvar="DATASET_DIR", show_default=True)
    ] = Path("dataset"),
    site_key: Annotated[str, typer.Option(help="Site key", envvar="SITE_KEY")] = DEFAULT_SITE_KEY,
    max_loop_count: Annotated[
        int, typer.Option(help="Maximum loop count", envvar="MAX_LOOP_COUNT")
    ] = 30,
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

    typer.echo(f"Starting collector - Config: {config}")

    # Launch collector
    try:
        asyncio.run(launch_collector(collector_config=config, headless=headless, locale=locale))
    except KeyboardInterrupt:
        typer.echo("Collector stopped")
        sys.exit(0)
    except Exception as e:
        typer.echo(f"Collector error: {e}")
        sys.exit(1)
