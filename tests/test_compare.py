"""Two-period comparison: the tag table, the pair table and their refusals."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from conftest import EngRange, Role, make_meta
from tsdive.compare import (
    BOOTSTRAP_REPLICATES,
    compare,
    parse_periods,
    read_periods,
    tag_changes,
)
from tsdive.errors import InsufficientQuality

BEFORE = "2024-03-01T00:00:00Z/2024-03-01T02:00:00Z"
AFTER = "2024-03-01T02:00:00Z/2024-03-01T04:00:00Z"
N = 241  # 60 s samples over both periods, inclusive of both bounds
SPLIT = 120  # first sample of the after period


def _frame(values, qualities=None) -> pd.DataFrame:
    base = pd.Timestamp("2024-03-01 00:00:00+00:00")
    return pd.DataFrame(
        {
            "timestamp": [base + pd.Timedelta(60 * i, unit="s") for i in range(len(values))],
            "value": values,
            "quality": qualities or ["GOOD"] * len(values),
        }
    )


def _tag(archive_factory, point_id, values, qualities=None, **meta_kwargs):
    return archive_factory(
        _frame(values, qualities),
        make_meta(point_id=point_id, sample_rate_s=60.0, **meta_kwargs),
    )


def _noisy(seed: int, shift: float = 0.0, level: float = 50.0) -> list[float]:
    """A tag wobbling around ``level``, stepped by ``shift`` in the after period."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 1.0, size=N)
    return [level + noise[i] + (shift if i >= SPLIT else 0.0) for i in range(N)]


def _walkers(seed: int, *, decouple: bool) -> tuple[list[float], list[float]]:
    """Two tags on one random walk; the second takes its own after the split."""
    rng = np.random.default_rng(seed)
    shared = 50.0 + np.cumsum(rng.normal(0.0, 1.0, size=N))
    other = shared + rng.normal(0.0, 0.02, size=N)
    if decouple:
        own = other[SPLIT] + np.cumsum(rng.normal(0.0, 1.0, size=N - SPLIT))
        other[SPLIT:] = own
    return [float(v) for v in shared], [float(v) for v in other]


def _row(result, point_id: str):
    return next(t for t in result.tags if t.label == point_id)


def test_parse_periods_refuses_an_overlap():
    with pytest.raises(ValueError, match="--before and --after overlap"):
        parse_periods(BEFORE, "2024-03-01T01:00:00Z/2024-03-01T04:00:00Z")


def test_parse_periods_refuses_the_periods_the_wrong_way_round():
    with pytest.raises(ValueError, match="state the earlier period as --before"):
        parse_periods(AFTER, BEFORE)


def test_the_boundary_sample_belongs_to_the_after_period(archive_factory):
    path = _tag(archive_factory, "FIC101.PV", _noisy(1))
    before, after = parse_periods(BEFORE, AFTER)
    read = read_periods([str(path)], before, after)[0]
    assert len(read.before.frame) == SPLIT
    assert len(read.after.frame) == N - SPLIT
    assert read.before.frame["timestamp"].max() == pd.Timestamp("2024-03-01T01:59:00Z")
    assert read.after.frame["timestamp"].min() == pd.Timestamp("2024-03-01T02:00:00Z")


def test_a_level_shift_is_measured_in_before_period_sigmas(archive_factory):
    quiet = _tag(archive_factory, "TIC101.PV", _noisy(2))
    shifted = _tag(archive_factory, "FIC101.PV", _noisy(3, shift=8.0))
    result = compare([str(quiet), str(shifted)], BEFORE, AFTER)

    # The shifted tag ranks first, and 8 units of a ~1.0-sigma tag is ~8 sigma.
    assert result.tags[0].label == "FIC101.PV"
    assert 6.0 < result.tags[0].level_shift_sigma < 10.0
    assert abs(_row(result, "TIC101.PV").level_shift_sigma) < 1.0
    assert result.tags[0].level_shift_reason is None


def test_the_spread_ratio_reports_a_widened_after_period(archive_factory):
    rng = np.random.default_rng(11)
    values = [
        50.0 + float(rng.normal(0.0, 4.0 if i >= SPLIT else 1.0)) for i in range(N)
    ]
    path = _tag(archive_factory, "FIC101.PV", values)
    other = _tag(archive_factory, "TIC101.PV", _noisy(12))
    row = _row(compare([str(path), str(other)], BEFORE, AFTER), "FIC101.PV")
    assert 2.5 < row.spread_ratio < 6.0


def test_a_flat_before_period_refuses_the_level_shift_and_the_spread(archive_factory):
    flat = _tag(archive_factory, "FIC101.PV", [50.0] * SPLIT + _noisy(4)[SPLIT:])
    other = _tag(archive_factory, "TIC101.PV", _noisy(5))
    row = _row(compare([str(flat), str(other)], BEFORE, AFTER), "FIC101.PV")
    assert row.level_shift_sigma is None
    assert row.spread_ratio is None
    assert row.level_shift_reason == "no spread before"
    assert row.spread_reason == "no spread before"


def test_the_flagged_share_screens_the_after_period_on_the_before_baseline(
    archive_factory,
):
    shifted = _tag(archive_factory, "FIC101.PV", _noisy(6, shift=8.0))
    quiet = _tag(archive_factory, "TIC101.PV", _noisy(7))
    result = compare([str(shifted), str(quiet)], BEFORE, AFTER)
    assert _row(result, "FIC101.PV").flagged_fraction == 1.0
    assert _row(result, "TIC101.PV").flagged_fraction < 0.05


def test_a_censored_before_period_refuses_the_flagged_share(archive_factory):
    pinned = _tag(
        archive_factory,
        "FIC101.PV",
        [100.0 if 10 <= i < 20 else 50.0 + (i % 7) for i in range(N)],
        eng_range=EngRange(zero=0.0, span=100.0),
    )
    other = _tag(archive_factory, "TIC101.PV", _noisy(8))
    row = _row(compare([str(pinned), str(other)], BEFORE, AFTER), "FIC101.PV")
    assert row.flagged_fraction is None
    assert row.flagged_reason.startswith("InsufficientQuality: ")
    assert "may never serve as a baseline" in row.flagged_reason


def test_a_short_before_period_refuses_the_flagged_share(archive_factory):
    path = _tag(archive_factory, "FIC101.PV", _noisy(9))
    other = _tag(archive_factory, "TIC101.PV", _noisy(10))
    row = _row(
        compare(
            [str(path), str(other)],
            "2024-03-01T00:00:00Z/2024-03-01T00:10:00Z",
            AFTER,
        ),
        "FIC101.PV",
    )
    assert row.flagged_fraction is None
    assert row.flagged_reason == (
        "InsufficientQuality: only 11 GOOD history samples; 30 required for a baseline"
    )


def test_changed_at_is_the_first_break_inside_the_after_period(archive_factory):
    shifted = _tag(archive_factory, "FIC101.PV", _noisy(13, shift=8.0))
    other = _tag(archive_factory, "TIC101.PV", _noisy(14))
    row = _row(compare([str(shifted), str(other)], BEFORE, AFTER), "FIC101.PV")
    assert row.changed_at == pd.Timestamp("2024-03-01T02:00:00Z")
    assert row.changed_at_reason is None


def test_a_steady_tag_has_no_break_inside_the_after_period(archive_factory):
    quiet = _tag(archive_factory, "FIC101.PV", _noisy(15))
    other = _tag(archive_factory, "TIC101.PV", _noisy(16))
    row = _row(compare([str(quiet), str(other)], BEFORE, AFTER), "FIC101.PV")
    assert row.changed_at is None
    assert row.changed_at_reason is None


def test_a_state_valued_tag_refuses_every_number_and_keeps_its_row(archive_factory):
    modes = _tag(
        archive_factory,
        "FIC101.MODE",
        ["R0" if i < SPLIT else "R1" for i in range(N)],
        role=Role.MODE,
    )
    other = _tag(archive_factory, "TIC101.PV", _noisy(17))
    result = compare([str(modes), str(other)], BEFORE, AFTER)
    row = _row(result, "FIC101.MODE")
    assert row.level_shift_sigma is None
    assert row.level_shift_reason == "no GOOD rows before"
    assert row.flagged_reason.startswith("InsufficientQuality: ")
    assert row.changed_at_reason.startswith("ValueError: ")
    assert "state-valued" in row.changed_at_reason
    # A MODE tag never reaches the alignment, so the pair table refuses.
    assert result.pairs.reason.startswith("1 of 2 tags carry GOOD numeric rows")


def test_a_bad_quality_share_replaces_ok(archive_factory):
    qualities = ["GOOD"] * SPLIT + ["COMM FAILURE" if i % 2 else "GOOD" for i in range(N - SPLIT)]
    path = _tag(archive_factory, "FIC101.PV", _noisy(18), qualities)
    other = _tag(archive_factory, "TIC101.PV", _noisy(19))
    row = _row(compare([str(path), str(other)], BEFORE, AFTER), "FIC101.PV")
    assert row.quality == "BAD 50%"
    assert row.quality_ok is False


def test_an_unmapped_quality_share_reports_uncertain(archive_factory):
    qualities = ["GOOD"] * SPLIT + ["SENSOR DRIFT"] * (N - SPLIT)
    path = _tag(archive_factory, "FIC101.PV", _noisy(20), qualities)
    other = _tag(archive_factory, "TIC101.PV", _noisy(21))
    row = _row(compare([str(path), str(other)], BEFORE, AFTER), "FIC101.PV")
    assert row.quality == "UNCERTAIN 100%"


def test_a_censored_after_period_reports_censored(archive_factory):
    values = [50.0 + (i % 7) for i in range(SPLIT)] + [100.0] * (N - SPLIT)
    path = _tag(
        archive_factory, "FIC101.PV", values, eng_range=EngRange(zero=0.0, span=100.0)
    )
    other = _tag(archive_factory, "TIC101.PV", _noisy(22))
    row = _row(compare([str(path), str(other)], BEFORE, AFTER), "FIC101.PV")
    assert row.quality == "censored"


def test_an_after_period_that_stopped_moving_reports_frozen(archive_factory):
    values = _noisy(23)[:SPLIT] + [55.0] * (N - SPLIT)
    path = _tag(archive_factory, "FIC101.PV", values)
    other = _tag(archive_factory, "TIC101.PV", _noisy(24))
    row = _row(compare([str(path), str(other)], BEFORE, AFTER), "FIC101.PV")
    assert row.quality == "frozen"


def test_an_unreadable_archive_keeps_a_row_and_the_others_still_answer(
    archive_factory, tmp_path
):
    good = _tag(archive_factory, "FIC101.PV", _noisy(25))
    other = _tag(archive_factory, "TIC101.PV", _noisy(26))
    missing = tmp_path / "plant1" / "PIC101.PV.parquet"
    result = compare([str(missing), str(good), str(other)], BEFORE, AFTER)
    refused = result.tags[0]
    assert refused.quality == "[FileNotFoundError]"
    assert refused.refused is True
    assert result.n_refused == 1
    assert (
        refused.level_shift_sigma,
        refused.spread_ratio,
        refused.flagged_fraction,
        refused.changed_at,
    ) == (None, None, None, None)
    assert len(result.tags) == 3
    assert _row(result, "FIC101.PV").flagged_fraction is not None


def test_every_archive_unreadable_refuses_the_command(tmp_path):
    missing = [str(tmp_path / f"{name}.parquet") for name in ("a", "b")]
    with pytest.raises(InsufficientQuality, match="none readable over both periods"):
        compare(missing, BEFORE, AFTER)


def test_the_tag_label_drops_the_source_when_every_archive_shares_one(
    archive_factory, tmp_path
):
    shared = compare(
        [
            str(_tag(archive_factory, "FIC101.PV", _noisy(27))),
            str(_tag(archive_factory, "TIC101.PV", _noisy(28))),
        ],
        BEFORE,
        AFTER,
    )
    assert {t.label for t in shared.tags} == {"FIC101.PV", "TIC101.PV"}
    assert shared.source_id == "plant1"

    mixed = compare(
        [
            str(_tag(archive_factory, "FIC101.PV", _noisy(29))),
            str(
                archive_factory(
                    _frame(_noisy(30)),
                    make_meta(source_id="plant2", point_id="TIC101.PV", sample_rate_s=60.0),
                )
            ),
        ],
        BEFORE,
        AFTER,
    )
    assert {t.label for t in mixed.tags} == {"plant1:FIC101.PV", "plant2:TIC101.PV"}
    assert mixed.source_id is None


def test_a_decoupled_pair_clears_the_bootstrap_interval(archive_factory):
    left, right = _walkers(31, decouple=True)
    paths = [
        str(_tag(archive_factory, "FIC101.PV", left)),
        str(_tag(archive_factory, "TIC101.PV", right)),
    ]
    table = compare(paths, BEFORE, AFTER).pairs
    assert table.reason is None
    assert table.n_pairs == 1
    pair = table.pairs[0]
    assert pair.pearson_before > 0.9
    assert abs(pair.pearson_after) < 0.5
    assert pair.delta < -0.5
    assert pair.hi < 0.0  # the interval excludes a delta of 0
    assert pair.clears is True
    assert table.rate_s == 60
    assert table.rate_source == "declared"


def test_a_pair_that_kept_moving_together_does_not_clear(archive_factory):
    left, right = _walkers(32, decouple=False)
    paths = [
        str(_tag(archive_factory, "FIC101.PV", left)),
        str(_tag(archive_factory, "TIC101.PV", right)),
    ]
    table = compare(paths, BEFORE, AFTER).pairs
    pair = table.pairs[0]
    assert pair.pearson_before > 0.9
    assert pair.pearson_after > 0.9
    assert pair.lo < 0.0 < pair.hi
    assert pair.clears is False
    assert table.clearing == ()


def test_the_pair_table_carries_spearman_beside_pearson(archive_factory):
    left, right = _walkers(33, decouple=True)
    paths = [
        str(_tag(archive_factory, "FIC101.PV", left)),
        str(_tag(archive_factory, "TIC101.PV", right)),
    ]
    pair = compare(paths, BEFORE, AFTER).pairs.pairs[0]
    assert pair.spearman_before > 0.9
    assert pair.spearman_delta < -0.5


def test_a_tag_frozen_through_both_periods_leaves_its_pairs_undefined(
    archive_factory,
):
    left, right = _walkers(34, decouple=False)
    paths = [
        str(_tag(archive_factory, "FIC101.PV", left)),
        str(_tag(archive_factory, "TIC101.PV", right)),
        str(_tag(archive_factory, "PIC101.PV", [42.0] * N)),
    ]
    table = compare(paths, BEFORE, AFTER).pairs
    assert table.n_pairs == 3
    assert table.n_undefined == 2
    assert len(table.pairs) == 1


def test_a_stepped_tag_keeps_its_delta_and_loses_its_interval(archive_factory):
    """A setpoint moves at a handful of samples, so most resamples miss them.

    The correlation over the whole period is still defined; the block
    bootstrap around it is not, because a resample that draws no step
    leaves the setpoint flat.
    """
    left, right = _walkers(43, decouple=True)
    steps = [40.0 + 15.0 * ((i // 40) % 3) for i in range(N)]
    paths = [
        str(_tag(archive_factory, "FIC101.PV", left)),
        str(_tag(archive_factory, "TIC101.PV", right)),
        str(_tag(archive_factory, "FIC101.SP", steps)),
    ]
    table = compare(paths, BEFORE, AFTER).pairs
    assert table.n_pairs == 3
    assert table.n_undefined == 0
    assert table.n_no_interval == 2
    stepped = [p for p in table.pairs if "FIC101.SP" in (p.left, p.right)]
    assert len(stepped) == 2
    for pair in stepped:
        assert np.isfinite(pair.delta)
        assert (pair.lo, pair.hi) == (None, None)
        assert pair.clears is False


def test_the_pair_table_refuses_archives_declaring_different_rates(archive_factory):
    left, right = _walkers(35, decouple=True)
    paths = [
        str(_tag(archive_factory, "FIC101.PV", left)),
        str(
            archive_factory(
                _frame(right), make_meta(point_id="TIC101.PV", sample_rate_s=30.0)
            )
        ),
    ]
    table = compare(paths, BEFORE, AFTER).pairs
    assert table.reason == (
        "archives declare different sample rates (plant1:FIC101.PV=60.0, "
        "plant1:TIC101.PV=30.0); pass --rate-s to state the grid"
    )
    assert table.pairs == ()


def test_a_stated_rate_replaces_the_declared_one(archive_factory):
    left, right = _walkers(36, decouple=True)
    paths = [
        str(_tag(archive_factory, "FIC101.PV", left)),
        str(
            archive_factory(
                _frame(right), make_meta(point_id="TIC101.PV", sample_rate_s=30.0)
            )
        ),
    ]
    table = compare(paths, BEFORE, AFTER, rate_s=60).pairs
    assert table.reason is None
    assert table.rate_source == "--rate-s"
    assert table.pairs[0].clears is True


def test_the_pair_table_refuses_a_grid_it_cannot_fill(archive_factory):
    left, right = _walkers(37, decouple=True)
    # Half the after period's rows are dropped, so the aligned grid holds
    # far less than PAIR_MIN_COVERAGE.
    thinned = [v for i, v in enumerate(right) if i < SPLIT or i % 4 == 0]
    frame = _frame(right).iloc[[i for i in range(N) if i < SPLIT or i % 4 == 0]]
    frame["value"] = thinned
    paths = [
        str(_tag(archive_factory, "FIC101.PV", left)),
        str(
            archive_factory(
                frame.reset_index(drop=True),
                make_meta(point_id="TIC101.PV", sample_rate_s=60.0),
            )
        ),
    ]
    table = compare(paths, BEFORE, AFTER).pairs
    assert table.reason.startswith("MspcAlignmentError: aligned coverage ")
    assert "refusing to interpolate across tags" in table.reason


def test_the_bootstrap_replays_identically(archive_factory):
    left, right = _walkers(38, decouple=True)
    paths = [
        str(_tag(archive_factory, "FIC101.PV", left)),
        str(_tag(archive_factory, "TIC101.PV", right)),
    ]
    first = compare(paths, BEFORE, AFTER).pairs.pairs[0]
    second = compare(paths, BEFORE, AFTER).pairs.pairs[0]
    assert (first.lo, first.hi) == (second.lo, second.hi)


def test_ten_tags_over_a_day_stay_inside_a_time_budget(archive_factory):
    """45 pairs x 200 resamples over 1,440-row periods, in seconds not minutes.

    The published budget is 40 tags (780 pairs) over 7 days at 60 s in
    under 20 s. This is the same code path an eighth of the size; a
    bootstrap that resampled per pair instead of per replicate would blow
    through it here first.
    """
    rng = np.random.default_rng(39)
    n = 2881  # 60 s samples over two 24 h periods
    shared = 50.0 + np.cumsum(rng.normal(0.0, 1.0, size=n))
    paths = []
    base = pd.Timestamp("2024-03-01 00:00:00+00:00")
    stamps = [base + pd.Timedelta(60 * i, unit="s") for i in range(n)]
    for k in range(10):
        values = shared + rng.normal(0.0, 0.3, size=n)
        paths.append(
            str(
                archive_factory(
                    pd.DataFrame(
                        {
                            "timestamp": stamps,
                            "value": [float(v) for v in values],
                            "quality": ["GOOD"] * n,
                        }
                    ),
                    make_meta(point_id=f"FIC{k:03d}.PV", sample_rate_s=60.0),
                )
            )
        )
    started = time.perf_counter()
    table = compare(
        paths,
        "2024-03-01T00:00:00Z/2024-03-02T00:00:00Z",
        "2024-03-02T00:00:00Z/2024-03-03T00:00:00Z",
    ).pairs
    elapsed = time.perf_counter() - started
    assert table.n_pairs == 45
    assert elapsed < 20.0, f"{elapsed:.1f}s for 45 pairs x {BOOTSTRAP_REPLICATES}"


def test_tag_changes_rank_refusals_and_quality_failures_first(archive_factory, tmp_path):
    missing = str(tmp_path / "plant1" / "ZIC101.PV.parquet")
    frozen = str(
        _tag(archive_factory, "PIC101.PV", _noisy(40)[:SPLIT] + [55.0] * (N - SPLIT))
    )
    shifted = str(_tag(archive_factory, "FIC101.PV", _noisy(41, shift=8.0)))
    quiet = str(_tag(archive_factory, "TIC101.PV", _noisy(42)))
    before, after = parse_periods(BEFORE, AFTER)
    rows = tag_changes(
        read_periods([quiet, shifted, frozen, missing], before, after),
        shared_source=True,
    )
    assert [r.label for r in rows[:2]] == ["PIC101.PV", "ZIC101.PV"]
    assert [r.quality_ok for r in rows] == [False, False, True, True]
    assert rows[2].label == "FIC101.PV"  # the largest shift among the ok tags
