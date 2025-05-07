import shutil
from pathlib import Path

from hcaptcha_challenger.agent.challenger import AgentConfig
from hcaptcha_challenger.helper.env_generator import generate_env_example


def test_env_generator():
    output_dir = Path(__file__).parent
    output_file = generate_env_example(AgentConfig, output_dir=output_dir)

    for dst_name in ["examples", "docker"]:
        dst_dir = output_dir.parent.joinpath(dst_name)
        if dst_dir.is_dir():
            shutil.copyfile(output_file, dst_dir.joinpath(".env.example"))

    print(f"\nContent Preview ({output_file}):")
    print("-" * 60)
    with open(output_file, "r", encoding="utf-8") as f:
        print(f.read())
    print("-" * 60)

    print(f"\n.env.example file has been successfully generated to: {output_file}")
