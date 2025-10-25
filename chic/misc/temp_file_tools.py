"""
Temporary file utilities for chic
"""

from __future__ import annotations

import pathlib
import shutil
import tempfile
from contextlib import contextmanager


@contextmanager
def create_temp_folder(prefix=tempfile.template, suffix=''):
    '''
    Context manager that creates a temporary folder and deletes it after usage.

    After the suite finishes, the temporary folder and all its files and
    subfolders will be deleted.
    '''
    temp_folder = pathlib.Path(tempfile.mkdtemp(prefix=prefix, suffix=suffix))
    try:
        yield temp_folder
    finally:
        shutil.rmtree(str(temp_folder), ignore_errors=True)
