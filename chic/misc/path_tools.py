from __future__ import annotations

import contextlib
import datetime as datetime_module
import json
import os
import pathlib
import shlex
import socket
import subprocess
import sys
from typing import Any, Optional, TextIO

import yaml

import colorama


def posh_path(path: pathlib.Path) -> str:
    """Get posh representation of path if available, otherwise return posix path."""
    posh_script_path = pathlib.Path(os.path.expandvars('$DX/bin/Common/posh'))
    if posh_script_path.exists():
        try:
            result = subprocess.run(
                ['python', str(posh_script_path), str(path)],
                check=True,
                stdout=subprocess.PIPE,
                encoding='utf-8'
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
        else:
            return result.stdout.strip()

    return path.as_posix()


