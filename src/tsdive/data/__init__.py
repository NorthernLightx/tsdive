"""Dataset utilities: synthetic backbone + stratified splits."""

from __future__ import annotations

from tsdive.data.backbone import FAULT_TYPES, Backbone, generate_backbone, load_truth
from tsdive.data.splits import (
    GroupSplit,
    StratifiedSplit,
    group_holdout,
    stratified_split,
)

__all__ = [
    "FAULT_TYPES",
    "Backbone",
    "GroupSplit",
    "StratifiedSplit",
    "generate_backbone",
    "group_holdout",
    "load_truth",
    "stratified_split",
]
