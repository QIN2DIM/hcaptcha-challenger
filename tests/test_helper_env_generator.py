from hcaptcha_challenger.agent.challenger import AgentConfig
from hcaptcha_challenger.helper.env_generator import generate_env_example


def test_env_generator():
    output_file = generate_env_example(AgentConfig)

    print(f"\nContent Preview ({output_file}):")
    print("-" * 60)
    with open(output_file, "r", encoding="utf-8") as f:
        print(f.read())
    print("-" * 60)

    print(f"\n.env.example file has been successfully generated to: {output_file}")
