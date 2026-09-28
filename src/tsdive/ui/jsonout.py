"""One JSON encoding for every ``--json`` payload.

:func:`to_jsonable` walks dataclasses, mappings, sequences, pandas
timestamps and numpy scalars into values ``json.dumps`` accepts.
``None`` stays ``null``, and so does a non-finite float, because JSON has
no NaN or Infinity. Timestamps come out as ISO 8601 in UTC, with the
``+00:00`` offset spelled out.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _timestamp(value: Any) -> str:
    stamp = pd.Timestamp(value)
    if stamp.tz is None:
        stamp = stamp.tz_localize("UTC")
    return stamp.tz_convert("UTC").isoformat()


def to_jsonable(obj: Any) -> Any:
    """``obj`` as JSON-serializable values, structure preserved."""
    if obj is None or isinstance(obj, bool | int | str):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, Enum):
        return to_jsonable(obj.value)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return to_jsonable(float(obj))
    if obj is pd.NaT:
        return None
    if isinstance(obj, pd.Timedelta):
        return obj.total_seconds()
    if isinstance(obj, pd.Timestamp | datetime):
        return _timestamp(obj)
    if isinstance(obj, date):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {
            f.name: to_jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)
        }
    if isinstance(obj, Mapping):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, np.ndarray | pd.Index | pd.Series):
        return [to_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, set | frozenset):
        return [to_jsonable(v) for v in sorted(obj, key=str)]
    if isinstance(obj, Sequence):
        return [to_jsonable(v) for v in obj]
    return str(obj)


def refusal_json(error: BaseException) -> dict[str, str]:
    """The object a typed refusal becomes on a JSON surface.

    ``tsdive <command> --json`` prints it on stdout, and the MCP server
    returns it as a tool result: ``result_kind`` is ``"refusal"``,
    ``error_type`` the class name and ``cause`` the message.

    Examples:
        >>> from tsdive.errors import InsufficientQuality
        >>> from tsdive.ui.jsonout import refusal_json
        >>> refusal_json(InsufficientQuality("censored"))
        {'result_kind': 'refusal', 'error_type': 'InsufficientQuality', 'cause': 'censored'}
    """
    return {
        "result_kind": "refusal",
        "error_type": type(error).__name__,
        "cause": str(error),
    }
