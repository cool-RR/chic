"""
Datetime utilities for chic
"""

import datetime as datetime_module


def format_timedelta(td: datetime_module.timedelta) -> str:
    total_seconds = int(td.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    if hours > 0:
        return f'{hours}h{minutes:02d}m{seconds:02d}s'.rjust(9)
    else:
        return f'{minutes:02d}m{seconds:02d}s'.rjust(9)
