"""
JSON utilities for chic - JSONLA reading/writing
"""

from __future__ import annotations

import json
import pathlib
from typing import Any, Mapping, Iterable, Iterator


class NumpyJsonEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy/jax arrays."""

    def default(self, obj: Any) -> Any:
        try:
            import numpy as np
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.floating):
                if np.isnan(obj):
                    return 'NaN'
                elif np.isinf(obj):
                    return 'Infinity' if obj > 0 else '-Infinity'
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
        except ImportError:
            pass

        try:
            from jax import numpy as jnp
            if isinstance(obj, jnp.ndarray):
                return obj.tolist()
        except ImportError:
            pass

        return super().default(obj)


class JsonlaWriter:
    """Writer for JSONLA format (JSON Lines Array)."""

    def __init__(self, path: pathlib.Path) -> None:
        self.path = path
        self._headers: list[str] | None = None
        self._headers_written = False
        self._checked_existing_file = False

    def write(self, row_or_rows: Mapping[str, Any] | Iterable[Mapping[str, Any]], /) -> None:
        """Write one or more rows to the JSONLA file."""
        # Parse input
        if isinstance(row_or_rows, Mapping):
            rows = [row_or_rows]
        else:
            rows = list(row_or_rows)

        if not rows:
            return

        # Check for existing file on first write
        if not self._checked_existing_file:
            self._checked_existing_file = True
            if self.path.exists() and self.path.stat().st_size > 0:
                # Read existing headers
                with self.path.open('r') as file:
                    first_line = file.readline().strip()
                    if first_line:
                        self._headers = json.loads(first_line)
                        self._headers_written = True

        # Initialize headers from first row if not set
        if self._headers is None:
            self._headers = list(rows[0].keys())

        # Write headers if not written yet
        if not self._headers_written:
            with self.path.open('a', encoding='utf-8') as file:
                json.dump(self._headers, file, cls=NumpyJsonEncoder)
                file.write('\n')
            self._headers_written = True

        # Process each row
        for row in rows:
            # Check for new headers
            new_keys = set(row.keys()) - set(self._headers)
            if new_keys:
                raise ValueError(f'New headers found: {sorted(new_keys)}')

            # Create row with all headers, filling missing values with None
            complete_row = [row.get(header) for header in self._headers]

            with self.path.open('a', encoding='utf-8') as file:
                json.dump(complete_row, file, cls=NumpyJsonEncoder)
                file.write('\n')


class JsonlaReader:
    """Reader for JSONLA format."""

    def __init__(self, path: pathlib.Path | str) -> None:
        self.path = pathlib.Path(path)
        self._headers: list[str] | None = None

    @property
    def headers(self) -> list[str]:
        if self._headers is None:
            with self.path.open('r', encoding='utf-8') as file:
                first_line = file.readline().strip()
                if not first_line:
                    raise ValueError(f'Empty file: {self.path}')
                self._headers = json.loads(first_line)
        return self._headers

    def __iter__(self) -> Iterator[dict[str, Any]]:
        with self.path.open('r', encoding='utf-8') as file:
            # Skip headers line
            try:
                next(file)
            except StopIteration:
                return

            # Read data rows
            for line in file:
                line = line.strip()
                if not line:
                    continue
                row_data = json.loads(line)
                yield dict(zip(self.headers, row_data))
