import sys

import typer

from hcaptcha_challenger.cli import dataset
from hcaptcha_challenger.utils import SiteKey

# Create top-level application
app = typer.Typer(
    name="hcaptcha-challenger",
    help="hCaptcha challenge tool",
    add_completion=False,
    invoke_without_command=True,  # Enable callback when no command is passed
)


@app.callback()
def main_callback(ctx: typer.Context):
    """
    Main callback. Shows help if no command is provided.
    """
    if ctx.invoked_subcommand is None:
        print(ctx.get_help())
        raise typer.Exit()


@app.command(name="help", help="Show help for a specific command.")
def help_command(
    ctx: typer.Context,
    command_path: list[str] = typer.Argument(
        None, help="The command path (e.g., 'dataset collect')."
    ),
):
    """
    Provides help for commands, similar to `command --help`.

    Example: hc help dataset collect
    """
    if not command_path:
        # If `hc help` is called with no arguments, show main help
        print(ctx.parent.get_help())
        raise typer.Exit()
    
    # Get the full command context to search through
    current_ctx = ctx.parent
    
    # Navigate through the command path to find the target command
    for i, cmd in enumerate(command_path):
        found = False
        
        # Try to find command in current context
        if hasattr(current_ctx.command, "commands"):
            for name, command in current_ctx.command.commands.items():
                if name == cmd:
                    # Create a new context for this command
                    current_ctx = typer.Context(command, parent=current_ctx, info_name=cmd)
                    found = True
                    break
        
        if not found:
            # If we didn't find it as a command, it might be a typer app
            # Use --help mechanism directly
            try:
                remaining_path = command_path[i:]
                print(f"Showing help for: {' '.join(remaining_path)}")
                cmd_list = [*sys.argv[0:1], *remaining_path, "--help"]
                app(cmd_list)
                return
            except SystemExit:
                # Typer will exit after showing help
                return
            except Exception:
                print(f"Error: Command '{cmd}' not found")
                raise typer.Exit(code=1)
    
    # Print help for the found command
    print(current_ctx.get_help())
    raise typer.Exit()


app.add_typer(dataset.app, name="dataset", help="Dataset collection tool")

DEFAULT_SITE_KEY = SiteKey.user_easy


def main():
    app()


if __name__ == "__main__":
    main()
