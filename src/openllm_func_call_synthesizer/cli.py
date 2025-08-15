"""Console script for openllm_func_call_synthesizer."""

import typer
from rich.console import Console

from openllm_func_call_synthesizer import utils

app = typer.Typer()
console = Console()


@app.command()
def main():
    """Console script for openllm_func_call_synthesizer."""
    console.print("Replace this message by putting your code into "
               "openllm_func_call_synthesizer.cli.main")
    console.print("See Typer documentation at https://typer.tiangolo.com/")
    utils.do_something_useful()


if __name__ == "__main__":
    app()
