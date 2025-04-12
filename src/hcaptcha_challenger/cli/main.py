import typer

from hcaptcha_challenger.cli import collect

# Create top-level application
app = typer.Typer(name="hcaptcha-challenger", help="hCaptcha challenge tool")

# Add subcommands to the top-level application
app.add_typer(collect.app, name="collect", help="Dataset collection tool")


def main():
    app()


if __name__ == "__main__":
    main()
