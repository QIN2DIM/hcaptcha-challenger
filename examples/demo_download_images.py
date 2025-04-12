import asyncio

from hcaptcha_challenger.cli import launch_collector, CollectorConfig


async def main():
    """
    uv sync --extra collector

    Returns:

    """
    collector_config = CollectorConfig()
    await launch_collector(collector_config, headless=False)


if __name__ == '__main__':
    asyncio.run(main())
