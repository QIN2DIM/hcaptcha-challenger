import typer

from hcaptcha_challenger.cli import dataset
from hcaptcha_challenger.utils import SiteKey

# Create top-level application
app = typer.Typer(name="hcaptcha-challenger", help="hCaptcha challenge tool", add_completion=False)

app.add_typer(dataset.app, name="dataset", help="Dataset collection tool")

DEFAULT_SITE_KEY = SiteKey.user_easy


def main():
    app()


if __name__ == "__main__":
    main()
