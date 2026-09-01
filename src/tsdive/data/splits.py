"""Provenance-stratified splits: the gate before any published number.

Assignment is a pure function of (stratum key, row index, seed) hashed
with SHA-256, so the same frame + seed always yields the same split on
any platform, and every stratum is represented in train/val/test.

Two splitters live here and they answer different questions.
:func:`stratified_split` cuts *rows* inside each stratum, which is right
when rows are independent and wrong when they are not. On a dataset whose
rows share a physical asset - a well, a compressor, a reactor - the same
asset lands on both sides of that cut and any score measures how well the
model memorised the asset's own signature. :func:`group_holdout` assigns
whole groups instead, and :meth:`GroupSplit.leakage_check` refuses a
split where one group reached two folds.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import pandas as pd

from tsdive.errors import GroupLeakage


@dataclass(frozen=True)
class StratifiedSplit:
    train: list[int]
    val: list[int]
    test: list[int]
    strata_counts: dict[str, dict[str, int]]

    def sizes(self) -> tuple[int, int, int]:
        return len(self.train), len(self.val), len(self.test)


def _rank(key: str, index: int) -> str:
    """Deterministic random-looking rank for one row within its stratum."""
    return hashlib.sha256(f"{key}|{index}".encode()).hexdigest()


def _largest_remainder(total: int, ratios: tuple[float, float, float]) -> tuple[int, int, int]:
    raw = [total * r for r in ratios]
    base = [int(x) for x in raw]
    remainder = total - sum(base)
    order = sorted(range(3), key=lambda i: raw[i] - base[i], reverse=True)
    for i in order[:remainder]:
        base[i] += 1
    return base[0], base[1], base[2]


def stratified_split(
    frame: pd.DataFrame,
    by: tuple[str, ...] = ("instance_source", "fault_type"),
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
) -> StratifiedSplit:
    """Split rows so every stratum of `by` is proportionally represented.

    Within a stratum, rows are ordered by a seeded SHA-256 rank and sliced
    into train/val/test, so proportions are exact (up to integer
    rounding), and the same frame + seed yields the identical split on any
    platform.
    """
    for col in by:
        if col not in frame.columns:
            raise KeyError(f"stratification column {col!r} missing")

    strata_rows: dict[str, list[int]] = {}
    for idx, row in enumerate(frame.itertuples(index=False)):
        d = dict(zip(frame.columns, row, strict=True))
        key = "|".join(str(d[c]) for c in by)
        strata_rows.setdefault(key, []).append(idx)

    buckets: list[list[int]] = [[], [], []]
    counts: dict[str, dict[str, int]] = {}
    for key, indices in strata_rows.items():
        ordered = sorted(indices, key=lambda i: _rank(f"{seed}:{key}", i))
        n_train, n_val, n_test = _largest_remainder(len(ordered), ratios)
        cuts = (
            ordered[:n_train],
            ordered[n_train : n_train + n_val],
            ordered[n_train + n_val : n_train + n_val + n_test],
        )
        for b, part in enumerate(cuts):
            buckets[b].extend(part)
        counts[key] = {
            "train": len(cuts[0]),
            "val": len(cuts[1]),
            "test": len(cuts[2]),
        }
    return StratifiedSplit(
        train=buckets[0], val=buckets[1], test=buckets[2], strata_counts=counts
    )


@dataclass(frozen=True)
class GroupSplit:
    """A k-fold split in which every group sits entirely inside one fold."""

    group_col: str
    stratum_col: str
    n_folds: int
    seed: int
    # group key -> fold index. One entry per group: the type makes a group
    # in two folds unrepresentable, and leakage_check proves the folds
    # list agrees with it.
    fold_of_group: dict[str, int]
    folds: tuple[tuple[str, ...], ...]
    fold_sizes: tuple[int, ...]
    strata_counts: tuple[dict[str, int], ...]
    group_sizes: dict[str, int]

    def groups(self, fold: int) -> tuple[str, ...]:
        return self.folds[fold]

    def test_indices(self, items: pd.DataFrame, fold: int) -> list[int]:
        held = set(self.folds[fold])
        return [
            i for i, g in enumerate(items[self.group_col].astype(str)) if g in held
        ]

    def train_indices(self, items: pd.DataFrame, fold: int) -> list[int]:
        held = set(self.folds[fold])
        return [
            i for i, g in enumerate(items[self.group_col].astype(str)) if g not in held
        ]

    def strata_present(self, fold: int) -> set[str]:
        return {s for s, n in self.strata_counts[fold].items() if n > 0}

    def leakage_check(self) -> None:
        """Refuse a split where any group reached more than one fold.

        Also refuses a fold list that disagrees with ``fold_of_group`` or
        that omits a known group: a split whose bookkeeping is wrong is
        not a split whose numbers may be published.
        """
        seen: dict[str, int] = {}
        for fold, members in enumerate(self.folds):
            for group in members:
                if group in seen:
                    raise GroupLeakage(
                        f"group {group!r} appears in fold {seen[group]} and fold {fold}; "
                        f"a number computed over this split would measure memorisation "
                        f"of {group!r}, not detection"
                    )
                seen[group] = fold
        disagreed = sorted(
            g for g, f in self.fold_of_group.items() if seen.get(g) != f
        )
        if disagreed:
            raise GroupLeakage(
                f"fold membership disagrees with fold_of_group for {disagreed}"
            )
        missing = sorted(set(seen) - set(self.fold_of_group))
        if missing:
            raise GroupLeakage(f"folds contain groups absent from fold_of_group: {missing}")


def _fold_cost(counts: list[dict[str, int]], strata: list[str]) -> float:
    """Spread of per-stratum counts across folds; lower is more balanced."""
    total = 0.0
    for s in strata:
        per_fold = [float(c.get(s, 0)) for c in counts]
        mean = sum(per_fold) / len(per_fold)
        total += math.sqrt(sum((x - mean) ** 2 for x in per_fold) / len(per_fold))
    return total


def group_holdout(
    items: pd.DataFrame,
    *,
    group_col: str,
    stratum_col: str,
    n_folds: int = 5,
    seed: int = 42,
) -> GroupSplit:
    """Assign whole groups to ``n_folds`` folds, balancing strata.

    Groups are taken largest first - ties broken by a seeded SHA-256 rank
    of the group key, so the order is identical on any platform - and each
    is placed in the fold that leaves the per-stratum counts least spread.
    A group is never divided: a well contributing 500 instances lands in
    exactly one fold, and the fold it lands in is 500 instances larger
    than its siblings. That imbalance is the shape of the data, and
    :attr:`GroupSplit.fold_sizes` reports it rather than hiding it behind
    a row-level cut.
    """
    for col in (group_col, stratum_col):
        if col not in items.columns:
            raise KeyError(f"column {col!r} missing")
    if n_folds < 2:
        raise ValueError(f"n_folds must be at least 2, got {n_folds}")

    groups = items[group_col].astype(str)
    strata = items[stratum_col].astype(str)
    per_group: dict[str, dict[str, int]] = {}
    sizes: dict[str, int] = {}
    for g, s in zip(groups, strata, strict=True):
        per_group.setdefault(g, {})
        per_group[g][s] = per_group[g].get(s, 0) + 1
        sizes[g] = sizes.get(g, 0) + 1
    if len(per_group) < n_folds:
        raise ValueError(
            f"{len(per_group)} group(s) cannot fill {n_folds} folds; "
            "a fold with no group holds nothing out"
        )

    stratum_keys = sorted(set(strata))
    # Descending size, seeded SHA-256 rank within a size tie: two groups of
    # equal size must not be ordered by their names, or the split would
    # track the naming scheme.
    order = sorted(per_group, key=lambda g: (-sizes[g], _rank(f"{seed}:{group_col}:{g}", 0)))

    counts: list[dict[str, int]] = [dict.fromkeys(stratum_keys, 0) for _ in range(n_folds)]
    totals = [0] * n_folds
    assignment: dict[str, int] = {}
    for group in order:
        best_fold = 0
        best_key: tuple[float, int, int] | None = None
        for f in range(n_folds):
            trial = [dict(c) for c in counts]
            for s, n in per_group[group].items():
                trial[f][s] = trial[f].get(s, 0) + n
            key = (_fold_cost(trial, stratum_keys), totals[f] + sizes[group], f)
            if best_key is None or key < best_key:
                best_key, best_fold = key, f
        assignment[group] = best_fold
        totals[best_fold] += sizes[group]
        for s, n in per_group[group].items():
            counts[best_fold][s] = counts[best_fold].get(s, 0) + n

    folds = tuple(
        tuple(sorted(g for g, f in assignment.items() if f == fold))
        for fold in range(n_folds)
    )
    split = GroupSplit(
        group_col=group_col,
        stratum_col=stratum_col,
        n_folds=n_folds,
        seed=seed,
        fold_of_group=dict(assignment),
        folds=folds,
        fold_sizes=tuple(totals),
        strata_counts=tuple(dict(c) for c in counts),
        group_sizes=dict(sizes),
    )
    split.leakage_check()
    return split
