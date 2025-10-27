from __future__ import annotations

import sys


def get_cute_cli_args() -> list[str]:
    args = sys.argv[1:]
    if args and (args[0] == 'chic'):
        del args[0]
    return args
