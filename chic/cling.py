#!/usr/bin/env python
"""
CLI entry point for chic package
"""

import click

from chic.lol import main as run_main
from chic.hp import main as hp_main


@click.group()
def cli():
    """
    chic - RL fine-tuning with reward-based optimization for Gemma3
    """
    pass


# Add commands
cli.add_command(run_main, name='run')
cli.add_command(hp_main, name='hp')


if __name__ == "__main__":
    cli()
