"""
Trek system for chic - structured logging and output management
Adapted from viola's trekking system
"""

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

from chic.json_tools import JsonlaWriter, JsonlaReader


# Constants
CHIC_HOME = pathlib.Path.home() / '.chic'
CONSOLE_OUTPUT_SENTINEL = '\x00CONSOLE_ONLY\x00'


def _get_now_string() -> str:
    """Get current timestamp as string for folder naming."""
    return datetime_module.datetime.now().isoformat().replace(':', '-').replace('.', '-').replace('T', '-')


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


class TeeStream:
    """Stream that writes to both console and file."""

    def __init__(self, original_stream: TextIO, path: pathlib.Path) -> None:
        self.original_stream = original_stream
        self.path = path

    def write(self, message: str) -> None:
        self.original_stream.write(message)
        if message.startswith(CONSOLE_OUTPUT_SENTINEL):
            return
        self.write_to_file(message)

    def write_to_file(self, message: str) -> None:
        with self.path.open('a', encoding='utf-8') as file:
            wrapped_file = colorama.initialise.wrap_stream(
                file, convert=None, strip=None, autoreset=False, wrap=True
            )
            try:
                wrapped_file.write(message)
            except UnicodeEncodeError:
                try:
                    wrapped_file.write(''.join(c for c in message if c.isascii()))
                except:
                    pass

    def flush(self) -> None:
        self.original_stream.flush()

    def close(self) -> None:
        """Close the stream (no-op, we don't actually close the original stream)."""
        pass


class Trek:
    """
    Main Trek class for managing training run folders and logging.

    Creates a timestamped folder at ~/.chic/{timestamp}/ and manages:
    - stdout/stderr redirection
    - JSONLA files for structured data
    - meta.yaml for run metadata
    """

    def __init__(
        self,
        folder: Optional[pathlib.Path | str] = None,
        parent_folder: pathlib.Path = CHIC_HOME
    ) -> None:
        """
        Initialize Trek.

        Args:
            folder: Existing folder path, or None to create new timestamped folder
            parent_folder: Parent directory for new Trek folders
        """
        if folder is None:
            # Create new timestamped folder
            folder = parent_folder / _get_now_string()
            folder.mkdir(parents=True, exist_ok=False)
        else:
            folder = pathlib.Path(folder).resolve()
            if not folder.exists():
                raise FileNotFoundError(folder)
            if not folder.is_dir():
                raise NotADirectoryError(folder)

        self.folder: pathlib.Path = folder

        # File paths
        self.stdout_path = self.folder / 'stdout.txt'
        self.stderr_path = self.folder / 'stderr.txt'
        self.meta_yaml_path = self.folder / 'meta.yaml'
        self.results_path = self.folder / 'results.jsonla'
        self.hyperparameters_path = self.folder / 'hyperparameters.jsonla'

        # Writers
        self.results_writer = JsonlaWriter(self.results_path)
        self.hyperparameters_writer = JsonlaWriter(self.hyperparameters_path)

        # For context manager
        self.original_stdout: Optional[TextIO] = None
        self.original_stderr: Optional[TextIO] = None

    def __repr__(self) -> str:
        return f'Trek({str(self.folder)!r})'

    def __str__(self) -> str:
        return str(self.folder)

    def __truediv__(self, string_or_pathlike: str | os.PathLike) -> pathlib.Path:
        return self.folder / string_or_pathlike

    @property
    def posh_folder_string(self) -> str:
        """Get posh representation of the Trek folder path."""
        return posh_path(self.folder)

    def write_meta(self, **kwargs) -> None:
        """Write or update meta.yaml with run metadata."""
        try:
            with self.meta_yaml_path.open('r') as yaml_file:
                existing_yaml = yaml.safe_load(yaml_file)
        except FileNotFoundError:
            existing_yaml = {}

        with self.meta_yaml_path.open('w') as yaml_file:
            yaml.dump(existing_yaml | kwargs, yaml_file)

    @property
    def meta(self) -> dict:
        """Load and return meta.yaml contents."""
        with self.meta_yaml_path.open('r') as file:
            return yaml.safe_load(file)

    def __enter__(self) -> Trek:
        """Enter context manager - redirect stdout/stderr and write metadata."""
        # Redirect stdout and stderr
        self.original_stdout = sys.stdout
        sys.stdout = TeeStream(self.original_stdout, self.stdout_path)

        self.original_stderr = sys.stderr
        sys.stderr = TeeStream(self.original_stderr, self.stderr_path)

        # Print trek path (using posh format)
        print(f'{colorama.Fore.LIGHTBLACK_EX}{colorama.Style.BRIGHT}{self.posh_folder_string}'
              f'{colorama.Style.RESET_ALL}')

        # Write tampino file for tmux integration
        tampino_path = pathlib.Path(os.environ.get('TAMPINO', '/tmp'))
        try:
            tmux_pane_guid = os.environ['TMUX_PANE_GUID']
        except KeyError:
            pass
        else:
            try:
                (tampino_path / f'trek_{tmux_pane_guid}').write_text(self.posh_folder_string)
            except Exception:
                pass

        # Write initial metadata
        self.write_meta(
            command_line=' '.join(map(shlex.quote, sys.argv)),
            startup_time=datetime_module.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            hostname=socket.gethostname(),
            path=str(self.folder),
        )

        return self

    def __exit__(self, exception_type, exception, traceback) -> None:
        """Exit context manager - restore stdout/stderr."""
        # Restore original streams
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr

        # Write exception info if any
        if exception is not None:
            import traceback as traceback_module
            traceback_text = ''.join(
                traceback_module.format_exception(exception_type, exception, traceback)
            )
            with self.stderr_path.open('a', encoding='utf-8') as stderr_file:
                stderr_file.write(traceback_text)

    @staticmethod
    def get_last(parent_folder: pathlib.Path = CHIC_HOME) -> Trek:
        """Get the most recently created Trek."""
        trek_folders = [folder for folder in parent_folder.iterdir() if folder.is_dir()]
        if not trek_folders:
            raise ValueError(f'No Trek folders found in {parent_folder}')
        return Trek(max(trek_folders))
