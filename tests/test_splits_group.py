"""Group holdout: whole wells on one side of the cut, or no number at all."""

from __future__ import annotations

import pandas as pd
import pytest

from tsdive.data import GroupSplit, group_holdout
from tsdive.errors import GroupLeakage


def items_frame(spec: dict[str, list[str]]) -> pd.DataFrame:
    """One row per instance from ``{group: [stratum, ...]}``."""
    rows = [
        {"instance": f"{group}_{i:04d}", "well": group, "folder_label": stratum}
        for group, strata in spec.items()
        for i, stratum in enumerate(strata)
    ]
    return pd.DataFrame(rows)


def toy_spec() -> dict[str, list[str]]:
    return {
        "WELL-A": ["0"] * 300 + ["4"] * 200,  # the 500-instance well
        "WELL-B": ["0"] * 40 + ["1"] * 10,
        "WELL-C": ["0"] * 30 + ["4"] * 20,
        "WELL-D": ["2"] * 12 + ["7"] * 8,
        "WELL-E": ["0"] * 9 + ["9"] * 6,
        "WELL-F": ["4"] * 7,
        "WELL-G": ["1"] * 5 + ["3"] * 5,
        "WELL-H": ["9"] * 4,
    }


def toy_split(seed: int = 42, n_folds: int = 5) -> GroupSplit:
    return group_holdout(
        items_frame(toy_spec()),
        group_col="well",
        stratum_col="folder_label",
        n_folds=n_folds,
        seed=seed,
    )


def test_no_group_crosses_folds():
    split = toy_split()
    seen: dict[str, int] = {}
    for fold, members in enumerate(split.folds):
        for well in members:
            assert well not in seen, f"{well} in folds {seen.get(well)} and {fold}"
            seen[well] = fold
    assert set(seen) == set(toy_spec())
    split.leakage_check()


def test_big_group_lands_in_exactly_one_fold():
    """A well with 500 instances is not spread across 5 folds."""
    items = items_frame(toy_spec())
    split = toy_split()
    holding = [f for f, members in enumerate(split.folds) if "WELL-A" in members]
    assert len(holding) == 1
    fold = holding[0]
    test_rows = items.iloc[split.test_indices(items, fold)]
    assert (test_rows["well"] == "WELL-A").sum() == 500
    # And its 500 rows are absent from that fold's training side.
    train_rows = items.iloc[split.train_indices(items, fold)]
    assert "WELL-A" not in set(train_rows["well"])
    # The fold it landed in is the large one; the split reports that
    # imbalance rather than hiding it.
    assert split.fold_sizes[fold] == max(split.fold_sizes)
    assert split.fold_sizes[fold] >= 500


def test_train_and_test_partition_every_row():
    items = items_frame(toy_spec())
    split = toy_split()
    for fold in range(split.n_folds):
        tr = split.train_indices(items, fold)
        te = split.test_indices(items, fold)
        assert set(tr).isdisjoint(te)
        assert sorted(tr + te) == list(range(len(items)))
        assert len(te) == split.fold_sizes[fold]


def test_deterministic_across_calls_and_sensitive_to_seed():
    a = toy_split(seed=42)
    b = toy_split(seed=42)
    assert a.folds == b.folds
    assert a.fold_of_group == b.fold_of_group
    c = toy_split(seed=7)
    assert set(c.fold_of_group) == set(a.fold_of_group)
    # Seeds only reorder equal-size groups, so a different seed may or may
    # not move a well; what it must never do is invent or lose one.
    assert sum(c.fold_sizes) == sum(a.fold_sizes)


def test_assignment_is_platform_stable():
    """Hash-based ordering, pinned: a platform that disagrees fails here."""
    split = toy_split(seed=42)
    assert split.fold_of_group == {
        "WELL-A": 0,
        "WELL-B": 1,
        "WELL-C": 2,
        "WELL-D": 3,
        "WELL-E": 4,
        "WELL-F": 3,
        "WELL-G": 4,
        "WELL-H": 3,
    }
    assert split.fold_sizes == (500, 50, 50, 31, 25)


def test_row_order_does_not_change_the_split():
    items = items_frame(toy_spec())
    shuffled = items.iloc[::-1].reset_index(drop=True)
    a = group_holdout(items, group_col="well", stratum_col="folder_label", seed=42)
    b = group_holdout(shuffled, group_col="well", stratum_col="folder_label", seed=42)
    assert a.fold_of_group == b.fold_of_group


def test_every_stratum_reaches_at_least_one_fold():
    split = toy_split()
    strata = {s for strata in toy_spec().values() for s in strata}
    reached = set().union(*(split.strata_present(f) for f in range(split.n_folds)))
    assert reached == strata
    for stratum in strata:
        total = sum(counts.get(stratum, 0) for counts in split.strata_counts)
        assert total == sum(
            strata_list.count(stratum) for strata_list in toy_spec().values()
        )


def test_stratum_carried_by_one_group_cannot_reach_more_than_one_fold():
    """Folder 3 lives only on WELL-G, so it can only ever be in one fold."""
    split = toy_split()
    holding = [f for f in range(split.n_folds) if split.strata_counts[f].get("3", 0)]
    assert len(holding) == 1
    assert split.fold_of_group["WELL-G"] == holding[0]


def test_leakage_check_refuses_a_duplicated_group():
    leaky = GroupSplit(
        group_col="well",
        stratum_col="folder_label",
        n_folds=2,
        seed=42,
        fold_of_group={"WELL-A": 0, "WELL-B": 1},
        folds=(("WELL-A", "WELL-B"), ("WELL-B",)),
        fold_sizes=(2, 1),
        strata_counts=({"0": 2}, {"0": 1}),
        group_sizes={"WELL-A": 1, "WELL-B": 1},
    )
    with pytest.raises(GroupLeakage, match="WELL-B"):
        leaky.leakage_check()


def test_missing_column_and_too_few_groups_refused():
    items = items_frame({"WELL-A": ["0"], "WELL-B": ["0"]})
    with pytest.raises(KeyError):
        group_holdout(items, group_col="nope", stratum_col="folder_label")
    with pytest.raises(ValueError, match="cannot fill"):
        group_holdout(items, group_col="well", stratum_col="folder_label", n_folds=5)
    with pytest.raises(ValueError, match="at least 2"):
        group_holdout(items, group_col="well", stratum_col="folder_label", n_folds=1)
