"""The clock control: a score that reads time and nothing else."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd


def clock_control(record_position: Mapping[Any, int] | pd.Series) -> dict[Any, float]:
    """Score every window by how far into its own record it sits.

    The control reads no sensor value. A record that opens normal and
    stays faulty once the fault arrives is separable by position alone,
    so the control is published beside every detector: the gap between a
    detector's AUC and this row's AUC is what the detector adds over time.

    ``record_position`` maps a window key to its position in its record,
    as a ``Mapping`` or a ``pandas.Series`` indexed by key. The result
    maps each key to ``float(position)`` in the same order.
    """
    return {key: float(position) for key, position in record_position.items()}
