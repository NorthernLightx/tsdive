"""Flatline detection that reports its signals instead of blending them.

Why value-repetition run-length is NOT used as a signal: healthy tags on
exception/compression historians emit periodic re-writes of an unchanged
value (compression heartbeats), so repeated identical values are the
*normal* signature of a steady compressed tag, not evidence of freezing.
Run-length would fire on exactly the healthy case this detector must not
falsely flag.

Signals (each reported independently, never summed into a score):

1. ``time_since_last_actual_change``: stall time since the value last
   differed, compared against the tag's own p99 of historical change
   intervals.
2. ``distinct_value_count_vs_p05``: distinct GOOD values in the window,
   compared against p05 of distinct counts over reference windows.
3. ``zero_std_while_coupled_tag_moves``: zero in-window standard
   deviation while an explicitly curated coupled tag is moving. Couplings
   come only from a caller-provided list; they are never inferred from
   correlation mining.

Clipping is consulted first: a sensor parked at full scale is saturated,
not frozen, and gets ``saturated_not_frozen`` rather than a flatline
verdict.

Three further cases return NOT ASSESSED rather than "no flatline", which
would assert health from no evidence
(see :class:`NotAssessed`): a ``role=MODE`` tag, whose stillness is the
plant holding a state rather than a sensor freezing; a window with no
valid samples; and a window where not one signal had anything to compare
against.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import cast

import numpy as np
import pandas as pd

from tsdive.store.identity import Role, TagIdentity
from tsdive.store.tagstore import Window


class NotAssessed(StrEnum):
    """Why a window carries no verdict. The value is the reported reason.

    ``fired=False`` means "assessed, and nothing fired". These are the
    cases where that sentence would assert health from no evidence, so
    the verdict is NOT ASSESSED instead.
    """

    NO_VALID_SAMPLES = "no valid samples in window"
    STATE_VALUED_TAG = "state-valued tag (role=MODE); stillness is not a flatline signal"
    NO_EVALUABLE_SIGNAL = "no evaluable signal (no reference history / no coupled tag)"


@dataclass(frozen=True)
class SignalFinding:
    signal: str
    fired: bool
    evidence: str
    # False when the signal had nothing to compare against - no reference
    # history, no reference windows, no curated coupling. An inevaluable
    # signal is not a passing one.
    evaluable: bool = True
    # The two numbers the evidence sentence states, so a caller reading
    # this signal never has to parse the prose back into floats: what
    # this window did, and what it was held against. Either is None when
    # the signal could not compute it.
    observed: float | None = None
    reference: float | None = None


@dataclass(frozen=True)
class FlatlineVerdict:
    identity: TagIdentity
    fired: bool
    saturated_not_frozen: bool = False
    signals: tuple[SignalFinding, ...] = field(default_factory=tuple)
    not_assessed: NotAssessed | None = None

    def summary_lines(self) -> list[str]:
        if self.saturated_not_frozen:
            return [
                "flatline verdict: NOT ASSESSED (window censored; saturated != frozen)"
            ]
        if self.not_assessed is not None:
            return [f"flatline verdict: NOT ASSESSED ({self.not_assessed})"]
        verdict = "FLATLINE SUSPECTED" if self.fired else "no flatline"
        lines = [f"flatline verdict: {verdict}"]
        for s in self.signals:
            mark = "FIRED" if s.fired else "ok   "
            lines.append(f"  [{mark}] {s.signal}: {s.evidence}")
        return lines


def _last_actual_change(frame: pd.DataFrame) -> pd.Timestamp | None:
    """Timestamp of the last GOOD sample whose value differed from the previous one.

    Values are compared as stored, never through ``float``: a MODE tag's
    states ("R0" -> "R1") are changes in exactly the same sense a
    measurement's are, and coercing them would raise instead.
    """
    good = frame[frame["valid"]]
    present = good[good["value"].notna()]
    if len(present) < 2:
        return None
    values = present["value"].reset_index(drop=True)
    stamps = present["timestamp"].reset_index(drop=True)
    # ``shift`` makes the first row compare against nothing, so it is
    # dropped rather than counted as a change.
    changed = values.ne(values.shift()).to_numpy(dtype=bool, na_value=False)[1:]
    at = np.flatnonzero(changed)
    if at.size == 0:
        return None
    return pd.Timestamp(stamps.iloc[int(at[-1]) + 1])


def _pctl(values: Sequence[float], q: float) -> float | None:
    if len(values) == 0:
        return None
    return float(np.quantile(np.asarray(values, dtype=float), q))


def _std_threshold(mean: float) -> float:
    return 1e-9 * max(1.0, abs(mean))


def assess_flatline(
    window: Window,
    *,
    reference_change_intervals_s: Sequence[float],
    reference_distinct_counts: Sequence[int],
    couplings: Sequence[tuple[TagIdentity, TagIdentity]] = (),
    coupled_windows: Mapping[TagIdentity, Window] | None = None,
) -> FlatlineVerdict:
    """Assess one window against the three signals above."""
    identity = window.identity
    if window.physics.clipping.censored:
        return FlatlineVerdict(
            identity=identity,
            fired=False,
            saturated_not_frozen=True,
            signals=(
                SignalFinding(
                    "clipping_consulted_first",
                    fired=False,
                    evidence=(
                        f"censored window (clipped fraction "
                        f"{window.physics.clipping.fraction}); saturation mimics flatline"
                    ),
                ),
            ),
        )

    if window.meta.role is Role.MODE:
        # A state tag holds one state until the plant changes it, so a
        # still window is what a settled unit looks like. Every signal
        # here reads stillness as evidence of a stuck sensor, which is
        # the wrong question to ask of a label.
        return FlatlineVerdict(
            identity=identity,
            fired=False,
            signals=(
                SignalFinding(
                    "state_valued_tag",
                    fired=False,
                    evidence=NotAssessed.STATE_VALUED_TAG.value,
                    evaluable=False,
                ),
            ),
            not_assessed=NotAssessed.STATE_VALUED_TAG,
        )

    good = window.frame[window.frame["valid"]]
    if len(good) == 0:
        # Nothing was observed, so neither "frozen" nor "moving" is
        # assertable. Signal 2 would otherwise fire on a distinct count of
        # zero and call an empty window a flatline.
        return FlatlineVerdict(
            identity=identity,
            fired=False,
            signals=(
                SignalFinding(
                    "valid_samples_present",
                    fired=False,
                    evidence=NotAssessed.NO_VALID_SAMPLES.value,
                    evaluable=False,
                ),
            ),
            not_assessed=NotAssessed.NO_VALID_SAMPLES,
        )

    findings: list[SignalFinding] = []
    fired = False

    # Signal 1: stall vs own p99 of change intervals.
    stall_s: float | None = None
    sig1_evaluable = False
    findings_stall = "no GOOD samples; signal not evaluable"
    if len(good) >= 2:
        last_change = _last_actual_change(window.frame)
        end = cast(pd.Timestamp, good["timestamp"].max())
        if last_change is None:
            stall_s = (
                end - cast(pd.Timestamp, good["timestamp"].min())
            ).total_seconds()
            findings_stall = (
                f"no change inside window at all (span {stall_s:.0f}s); "
                "cannot beat a p99 the tag never established"
            )
        else:
            stall_s = (end - last_change).total_seconds()
        p99 = _pctl(reference_change_intervals_s, 0.99)
        if p99 is not None and stall_s is not None:
            did_fire = stall_s > p99
            fired = fired or did_fire
            sig1_evaluable = True
            findings_stall = f"stall {stall_s:.0f}s vs own p99 {p99:.0f}s"
        elif p99 is None:
            findings_stall += "; no reference history provided"
    findings.append(
        SignalFinding(
            "time_since_last_actual_change",
            fired,
            findings_stall,
            evaluable=sig1_evaluable,
            observed=stall_s,
            reference=p99 if len(good) >= 2 else None,
        )
    )

    # Signal 2: distinct count vs reference p05.
    distinct = int(good["value"].dropna().nunique())
    p05 = _pctl([float(c) for c in reference_distinct_counts], 0.05)
    sig2_fired = p05 is not None and distinct < p05
    fired = fired or sig2_fired
    if p05 is None:
        ev2 = "no reference windows provided; signal not evaluable"
    else:
        ev2 = f"{distinct} distinct values vs reference p05 {p05:.1f}"
    findings.append(
        SignalFinding(
            "distinct_value_count_vs_p05",
            sig2_fired,
            ev2,
            evaluable=p05 is not None,
            observed=float(distinct),
            reference=p05,
        )
    )

    # Signal 3: zero std while a curated coupled tag moves.
    sig3_fired = False
    sig3_evaluable = False
    ev3 = "no curated coupling applies to this tag"
    std: float | None = None
    coupled_std: float | None = None
    values = good["value"].dropna().astype(float)
    if len(values) > 1:
        std = float(values.std())
        mean = float(values.mean())
        coupled_map = coupled_windows or {}
        for a, b in couplings:
            if a != identity or b not in coupled_map:
                continue
            other = coupled_map[b]
            ogood = other.frame[other.frame["valid"]]
            ovalues = ogood["value"].dropna().astype(float)
            if len(ovalues) <= 1:
                ev3 = f"coupled tag {b} lacks evaluable samples"
                continue
            ostd = float(ovalues.std())
            omoving = ostd > _std_threshold(float(ovalues.mean()))
            sig3_evaluable = True
            coupled_std = ostd
            if std <= _std_threshold(mean) and omoving:
                sig3_fired = True
                ev3 = (
                    f"std {std:.3g} ~ 0 while coupled tag {b} moves (std {ostd:.3g})"
                )
            else:
                ev3 = f"coupled tag {b} present; std {std:.3g} vs other std {ostd:.3g}"
    fired = fired or sig3_fired
    findings.append(
        SignalFinding(
            "zero_std_while_coupled_tag_moves",
            sig3_fired,
            ev3,
            evaluable=sig3_evaluable,
            observed=std,
            reference=coupled_std,
        )
    )

    if not any(s.evaluable for s in findings):
        # Nothing was compared against anything. "no flatline" here would
        # be a clean bill of health signed by no one.
        return FlatlineVerdict(
            identity=identity,
            fired=False,
            signals=tuple(findings),
            not_assessed=NotAssessed.NO_EVALUABLE_SIGNAL,
        )
    return FlatlineVerdict(identity=identity, fired=fired, signals=tuple(findings))
