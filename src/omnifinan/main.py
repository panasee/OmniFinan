"""Module entrypoint delegating to presentation CLI."""

from colorama import init
from pyomnix.omnix_logger import get_logger

from .presentation.cli import run_cli

logger = get_logger(__name__)
init(autoreset=True)


if __name__ == "__main__":
    run_cli()
