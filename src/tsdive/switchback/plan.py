"""A switchback plan: the schedule of settings A and B, its digest, and its power readout.

A plan cuts the window from ``start`` into K blocks of ``block_s``
seconds (K = floor(span / block), less one when odd), assigns exactly
K/2 of them to setting B with the seed, and marks the first
``washout_s`` seconds of every block as washout. The digest is SHA-256
over a canonical text of the plan's fields and blocks, written with
integers and ISO 8601 UTC timestamps only, so the same plan gives the
same digest on every platform.

:func:`verify_plan` recomputes the schedule from the fields and raises
:class:`~tsdive.errors.ScheduleMismatch` when the blocks, the digest, the
balance or the seed disagree. The power readout lays seeded schedules of
the same design over a history window where nothing was switched, adds
delta * sigma to the B blocks, and counts how often the 95% interval
excludes 0.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from tsdive.errors import ScheduleMismatch
from tsdive.switchback.design import (
    PERMUTATIONS,
    Blocks,
    Design,
    design_size,
    make_design,
    setting,
)
from tsdive.switchback.inference import (
    EMPTY_BLOCK,
    MAD_TO_SD,
    NO_SPREAD,
    assignment_terms,
    mad,
    make_frame,
    randomization,
)

PLAN_FORMAT = "tsdive-switchback-plan"
PLAN_VERSION = 1

SETTING_A = "A"
SETTING_B = "B"

POWER_DRAWS = 200
"""Seeded schedules the power readout draws over the history window."""
POWER_DELTAS = (0.0, 0.1, 0.25, 0.5, 1.0)
"""Shifts the power readout adds to the B blocks, in sigma (1.4826 MAD of the history)."""
POWER_TARGET = 0.8
HISTORY_SHORT = "history_short"

_NS = 1_000_000_000


def iso(stamp: pd.Timestamp) -> str:
    """``stamp`` in UTC as ISO 8601 with the ``+00:00`` offset: the digest's time format."""
    return cast(pd.Timestamp, pd.Timestamp(stamp)).tz_convert("UTC").isoformat()


@dataclass(frozen=True)
class PlannedBlock:
    """One block of the schedule: when it runs, its setting, and when its washout ends."""

    index: int
    start: pd.Timestamp
    end: pd.Timestamp
    setting: str
    washout_end: pd.Timestamp


@dataclass(frozen=True)
class PowerReadout:
    """Detection rates of the plan's design over a history window where nothing changed.

    ``rates[i]`` is the share of ``draws`` seeded schedules whose 95%
    interval excludes 0 after ``deltas[i]`` * ``sigma`` is added to the B
    blocks: the claim rate at a zero shift, detection above it.
    ``smallest`` is the smallest shift on the grid, in sigma, detected at
    ``POWER_TARGET`` or more, or ``None``. On a refusal ``reason`` names
    it and the rates are empty.
    """

    tag: str
    start: pd.Timestamp
    end: pd.Timestamp
    unit: str | None
    draws: int
    deltas: tuple[float, ...]
    sigma: float | None = None
    n_kept: int = 0
    rates: tuple[float, ...] = ()
    smallest: float | None = None
    reason: str | None = None
    detail: str | None = None


@dataclass(frozen=True)
class SwitchbackPlan:
    """A balanced random schedule of settings A and B over one window.

    ``start`` and ``end`` are the requested window; the schedule runs from
    ``start`` to :attr:`schedule_end`, K whole blocks. ``min_p`` is the
    smallest two-sided p-value the design can reach. Build one with
    :func:`make_plan` or :func:`tsdive.api.switchback_plan`.
    """

    start: pd.Timestamp
    end: pd.Timestamp
    block_s: int
    washout_s: int
    seed: int
    blocks: tuple[PlannedBlock, ...]
    n_assignments: int
    enumerated: bool
    min_p: float
    digest: str
    power: PowerReadout | None = None

    @property
    def k(self) -> int:
        return len(self.blocks)

    @property
    def schedule_end(self) -> pd.Timestamp:
        return cast(pd.Timestamp, self.start + pd.Timedelta(self.k * self.block_s, unit="s"))

    @property
    def observed(self) -> np.ndarray:
        """(K,) int8, 1 marks a B block."""
        return np.array([b.setting == SETTING_B for b in self.blocks], dtype=np.int8)

    @property
    def reference_size(self) -> int:
        """Assignments the p-value is computed over, the observed one included."""
        return self.n_assignments if self.enumerated else PERMUTATIONS + 1

    def render(self) -> str:
        """The text ``tsdive switchback plan`` prints, without its ``wrote`` line.

        Examples:
            >>> import tsdive
            >>> plan = tsdive.switchback_plan("2024-06-03T00:00:00Z", "2024-06-04T00:00:00Z",
            ...                               block="PT1H", washout="PT15M", seed=7)
            >>> print(plan.render().splitlines()[0])
            switchback plan   24 blocks of 1 h   A 12   B 12   digest 66da65ede04f
        """
        from tsdive.switchback.render import plan_lines

        return "\n".join(plan_lines(self))

    def to_dict(self) -> dict[str, object]:
        """The plan document: what :meth:`write_json` writes, ready for ``json.dumps``."""
        return plan_to_dict(self)

    def write_json(self, path: str | Path, *, overwrite: bool = False) -> Path:
        """Write the plan document to ``path``; an existing file raises ``FileExistsError``."""
        out = Path(path)
        if out.exists() and not overwrite:
            raise FileExistsError(
                f"{out.as_posix()} exists; a plan file records one randomization, so "
                "write the new plan to another path"
            )
        text = json.dumps(self.to_dict(), indent=2) + "\n"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8", newline="\n")
        return out

    @classmethod
    def read_json(cls, path: str | Path) -> SwitchbackPlan:
        """Read a plan document. The schedule is checked by :func:`verify_plan`, not here.

        Raises:
            ValueError: the file is not a plan document of this version.
        """
        return plan_from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


# ---------------------------------------------------------------- digest


def canonical_schedule(
    start: pd.Timestamp,
    end: pd.Timestamp,
    block_s: int,
    washout_s: int,
    seed: int,
    blocks: tuple[PlannedBlock, ...],
) -> str:
    """The text the digest is taken over: integers and ISO 8601 UTC timestamps only."""
    lines = [
        f"{PLAN_FORMAT} {PLAN_VERSION}",
        f"start {iso(start)}",
        f"end {iso(end)}",
        f"block_s {int(block_s)}",
        f"washout_s {int(washout_s)}",
        f"seed {int(seed)}",
    ]
    lines.extend(
        f"block {b.index} {iso(b.start)} {iso(b.end)} {iso(b.washout_end)} {b.setting}"
        for b in blocks
    )
    return "\n".join(lines)


def plan_digest(plan: SwitchbackPlan) -> str:
    """SHA-256 hex digest of the plan's canonical schedule.

    Examples:
        >>> import pandas as pd
        >>> from tsdive.switchback import make_plan, plan_digest
        >>> plan = make_plan(pd.Timestamp("2024-06-03T00:00:00Z"),
        ...                  pd.Timestamp("2024-06-04T00:00:00Z"),
        ...                  block_s=3600, washout_s=900, seed=7)
        >>> plan_digest(plan)[:12], plan_digest(plan) == plan.digest
        ('66da65ede04f', True)
    """
    text = canonical_schedule(
        plan.start, plan.end, plan.block_s, plan.washout_s, plan.seed, plan.blocks
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------- plan


def _block_count(start: pd.Timestamp, end: pd.Timestamp, block_s: int) -> int:
    span_ns = int((end - start).value)
    k = span_ns // (block_s * _NS)
    return int(k - (k % 2))


def _blocks(
    start: pd.Timestamp, block_s: int, washout_s: int, observed: np.ndarray
) -> tuple[PlannedBlock, ...]:
    length = pd.Timedelta(block_s, unit="s")
    washout = pd.Timedelta(washout_s, unit="s")
    return tuple(
        PlannedBlock(
            index=i,
            start=cast(pd.Timestamp, start + i * length),
            end=cast(pd.Timestamp, start + (i + 1) * length),
            setting=SETTING_B if observed[i] else SETTING_A,
            washout_end=cast(pd.Timestamp, start + i * length + washout),
        )
        for i in range(len(observed))
    )


def make_plan(
    start: pd.Timestamp, end: pd.Timestamp, block_s: int, washout_s: int, seed: int
) -> SwitchbackPlan:
    """The balanced schedule ``seed`` draws over ``start`` to ``end``.

    Raises:
        ValueError: a naive bound, ``end`` not after ``start``, a block
            of less than one second, a washout outside [0, block), or a
            negative seed.
        DesignTooSmall: the window holds too few blocks for a
            randomization test at the 5% level.

    Examples:
        >>> import pandas as pd
        >>> from tsdive.switchback import make_plan
        >>> plan = make_plan(pd.Timestamp("2024-06-03T00:00:00Z"),
        ...                  pd.Timestamp("2024-06-04T00:00:00Z"),
        ...                  block_s=3600, washout_s=900, seed=7)
        >>> plan.k, plan.blocks[0].setting, plan.blocks[0].washout_end.isoformat()
        (24, 'A', '2024-06-03T00:15:00+00:00')
    """
    for name, stamp in (("start", start), ("end", end)):
        if stamp.tz is None:
            raise ValueError(f"plan {name} {stamp} is naive; state it in UTC")
    start = cast(pd.Timestamp, start.tz_convert("UTC"))
    end = cast(pd.Timestamp, end.tz_convert("UTC"))
    if end <= start:
        raise ValueError("plan end is not after its start")
    if block_s < 1:
        raise ValueError(f"block of {block_s} s; a block lasts at least 1 s")
    if not 0 <= washout_s < block_s:
        raise ValueError(
            f"washout of {washout_s} s with blocks of {block_s} s; the washout must be "
            "0 or more and shorter than a block"
        )
    if seed < 0:
        raise ValueError(f"seed {seed} is negative; pass a seed of 0 or more")
    k = _block_count(start, end, block_s)
    design = make_design(k, seed)
    blocks = _blocks(start, block_s, washout_s, design.observed)
    plan = SwitchbackPlan(
        start=start,
        end=end,
        block_s=block_s,
        washout_s=washout_s,
        seed=seed,
        blocks=blocks,
        n_assignments=design.n_assignments,
        enumerated=design.enumerated,
        min_p=design.min_p,
        digest="",
    )
    return replace(plan, digest=plan_digest(plan))


def verify_plan(plan: SwitchbackPlan) -> Design:
    """The design the plan's seed draws, after checking the plan against it.

    Checked in this order: the digest against the canonical schedule, each
    block's index and times against ``start``, ``block_s`` and
    ``washout_s``, the balance of the settings, and the settings against
    the ones the seed draws.

    Raises:
        ScheduleMismatch: any check fails.
        DesignTooSmall: the plan's block count is too small for a
            randomization test.

    Examples:
        >>> import dataclasses
        >>> import pandas as pd
        >>> from tsdive.switchback import make_plan, verify_plan
        >>> plan = make_plan(pd.Timestamp("2024-06-03T00:00:00Z"),
        ...                  pd.Timestamp("2024-06-04T00:00:00Z"),
        ...                  block_s=3600, washout_s=900, seed=7)
        >>> design = verify_plan(plan)
        >>> design.k, bool((design.observed == plan.observed).all())
        (24, True)
        >>> verify_plan(dataclasses.replace(plan, seed=8))
        Traceback (most recent call last):
        ...
        tsdive.errors.ScheduleMismatch: ...
    """
    actual = plan_digest(plan)
    if actual != plan.digest:
        raise ScheduleMismatch(
            f"plan digest {plan.digest[:12]} does not match its schedule "
            f"({actual[:12]}); the plan was edited after it was written, so plan "
            "the trial again"
        )
    k = _block_count(plan.start, plan.end, plan.block_s)
    if plan.k != k:
        raise ScheduleMismatch(
            f"the plan lists {plan.k} blocks; its window and block length give {k}"
        )
    expected = _blocks(plan.start, plan.block_s, plan.washout_s, np.zeros(k, dtype=np.int8))
    for got, want in zip(plan.blocks, expected, strict=True):
        if (got.index, got.start, got.end, got.washout_end) != (
            want.index,
            want.start,
            want.end,
            want.washout_end,
        ):
            raise ScheduleMismatch(
                f"block {got.index} runs {iso(got.start)} to {iso(got.end)}; the plan's "
                f"start and block length put block {want.index} at {iso(want.start)} "
                f"to {iso(want.end)}"
            )
        if got.setting not in (SETTING_A, SETTING_B):
            raise ScheduleMismatch(
                f"block {got.index} has setting {got.setting!r}; a setting is A or B"
            )
    n_b = int(plan.observed.sum())
    if n_b * 2 != k:
        raise ScheduleMismatch(
            f"the schedule is not balanced: {n_b} of {k} blocks run setting B, and a "
            f"balanced schedule runs {k // 2}"
        )
    design = make_design(k, plan.seed)
    if not np.array_equal(design.observed, plan.observed):
        first = int(np.flatnonzero(design.observed != plan.observed)[0])
        raise ScheduleMismatch(
            f"block {first} runs setting {plan.blocks[first].setting}, and seed "
            f"{plan.seed} assigns it the other setting"
        )
    return design


# ---------------------------------------------------------------- JSON


def json_count(n: int) -> int | None:
    """``n`` when a JSON reader holds it exactly (at most 2**53), else ``None``."""
    return n if n.bit_length() <= 53 else None


def _power_to_dict(power: PowerReadout) -> dict[str, object]:
    sigma = power.sigma
    return {
        "tag": power.tag,
        "history": {"start": iso(power.start), "end": iso(power.end)},
        "unit": power.unit,
        "draws": power.draws,
        "sigma": sigma,
        "kept": power.n_kept,
        "rows": [
            {
                "shift_sigma": delta,
                "shift": None if sigma is None else delta * sigma,
                "claim_rate": rate,
            }
            for delta, rate in zip(power.deltas, power.rates, strict=False)
        ],
        "detection_target": POWER_TARGET,
        "smallest_shift_sigma": power.smallest,
        "smallest_shift": (
            None if power.smallest is None or sigma is None else power.smallest * sigma
        ),
        "reason": power.reason,
        "detail": power.detail,
    }


def plan_to_dict(plan: SwitchbackPlan) -> dict[str, object]:
    """The plan document: integers, floats, strings and ISO 8601 UTC timestamps."""
    return {
        "format": PLAN_FORMAT,
        "version": PLAN_VERSION,
        "window": {"start": iso(plan.start), "end": iso(plan.end)},
        "block_s": plan.block_s,
        "washout_s": plan.washout_s,
        "seed": plan.seed,
        "blocks": plan.k,
        "schedule_end": iso(plan.schedule_end),
        "assignments": json_count(plan.n_assignments),
        "reference": "enumerated" if plan.enumerated else "sampled",
        "reference_size": plan.reference_size,
        "smallest_p": plan.min_p,
        "digest": plan.digest,
        "schedule": [
            {
                "block": b.index,
                "start": iso(b.start),
                "end": iso(b.end),
                "setting": b.setting,
                "washout_end": iso(b.washout_end),
            }
            for b in plan.blocks
        ],
        "power": None if plan.power is None else _power_to_dict(plan.power),
    }


def _stamp(text: object, key: str) -> pd.Timestamp:
    if not isinstance(text, str):
        raise ValueError(f"plan field {key!r} must be an ISO 8601 string")
    stamp = cast(pd.Timestamp, pd.Timestamp(text))
    if stamp.tz is None:
        raise ValueError(f"plan field {key!r} is naive ({text}); a plan stores UTC")
    return cast(pd.Timestamp, stamp.tz_convert("UTC"))


def _int(value: object, key: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"plan field {key!r} must be an integer")
    return value


def _power_from_dict(doc: object) -> PowerReadout | None:
    if doc is None:
        return None
    if not isinstance(doc, dict):
        raise ValueError("plan field 'power' must be an object or null")
    rows = doc.get("rows") or []
    history = doc.get("history") or {}
    return PowerReadout(
        tag=str(doc.get("tag")),
        start=_stamp(history.get("start"), "power.history.start"),
        end=_stamp(history.get("end"), "power.history.end"),
        unit=doc.get("unit"),
        draws=_int(doc.get("draws"), "power.draws"),
        deltas=tuple(float(r["shift_sigma"]) for r in rows) or POWER_DELTAS,
        sigma=doc.get("sigma"),
        n_kept=_int(doc.get("kept", 0), "power.kept"),
        rates=tuple(float(r["claim_rate"]) for r in rows),
        smallest=doc.get("smallest_shift_sigma"),
        reason=doc.get("reason"),
        detail=doc.get("detail"),
    )


def plan_from_dict(doc: object) -> SwitchbackPlan:
    """A plan from its document. Raises ``ValueError`` on a document of another kind."""
    if not isinstance(doc, dict) or doc.get("format") != PLAN_FORMAT:
        raise ValueError(f"not a switchback plan: the file carries no format {PLAN_FORMAT!r}")
    if doc.get("version") != PLAN_VERSION:
        raise ValueError(
            f"switchback plan version {doc.get('version')!r}; this tsdive reads "
            f"version {PLAN_VERSION}"
        )
    try:
        window = doc["window"]
        schedule = doc["schedule"]
        blocks = tuple(
            PlannedBlock(
                index=_int(b["block"], "schedule.block"),
                start=_stamp(b["start"], "schedule.start"),
                end=_stamp(b["end"], "schedule.end"),
                setting=str(b["setting"]),
                washout_end=_stamp(b["washout_end"], "schedule.washout_end"),
            )
            for b in schedule
        )
        start = _stamp(window["start"], "window.start")
        end = _stamp(window["end"], "window.end")
        block_s = _int(doc["block_s"], "block_s")
        washout_s = _int(doc["washout_s"], "washout_s")
        seed = _int(doc["seed"], "seed")
        digest = str(doc["digest"])
    except (KeyError, TypeError) as e:
        raise ValueError(f"switchback plan is missing field {e}") from e
    n, enumerated, min_p = design_size(len(blocks))
    return SwitchbackPlan(
        start=start,
        end=end,
        block_s=block_s,
        washout_s=washout_s,
        seed=seed,
        blocks=blocks,
        n_assignments=n,
        enumerated=enumerated,
        min_p=min_p,
        digest=digest,
        power=_power_from_dict(doc.get("power")),
    )


# ---------------------------------------------------------------- power


def schedule_offsets(
    times: pd.Series, start: pd.Timestamp, block_s: int, k: int
) -> Blocks:
    """Blocks of ``block_s`` seconds from ``start`` for UTC ``times``, in integer nanoseconds.

    ``offset`` and ``length`` are nanoseconds, so a washout compared with
    them is exact to the nanosecond. ``times`` stored at a coarser
    resolution are converted to nanoseconds first.
    """
    elapsed = (pd.DatetimeIndex(times) - start).as_unit("ns").asi8
    length = block_s * _NS
    index = np.floor_divide(elapsed, length)
    offset = elapsed - index * length
    block = np.where((index >= 0) & (index < k), index, -1)
    return Blocks(k, float(length), block, offset)


def power_rates(
    y: np.ndarray, blocks: Blocks, washout_ns: int, seed: int, draws: int = POWER_DRAWS
) -> tuple[float, int, tuple[float, ...]] | str:
    """(sigma, kept samples, claim share per shift) over ``draws`` schedules, or a refusal.

    The shifts are ``POWER_DELTAS``. Draw j uses the design
    ``make_design(K, (seed, j))``. The shift is a
    step (no lag) of delta * sigma in every B block, sigma = 1.4826 MAD of
    the finite values of ``y``. A claim is a 95% interval excluding 0.
    Refusals: those of :func:`make_frame`, and ``empty_block``.
    """
    finite = y[np.isfinite(y)]
    sigma = MAD_TO_SD * mad(finite) if finite.size else 0.0
    if sigma == 0:
        return NO_SPREAD
    frame = make_frame(y, None, blocks, washout_ns)
    if isinstance(frame, str):
        return frame
    if not frame.nonempty.all():
        return EMPTY_BLOCK
    deltas = np.array(POWER_DELTAS)
    claims = np.zeros(len(deltas))
    for j in range(draws):
        design = make_design(blocks.k, (seed, j))
        terms = assignment_terms(frame, design)
        if isinstance(terms, str):
            return terms
        g = setting(blocks, design.observed)[frame.positions]
        g_sums = frame.sums(frame.project_out(g))
        sums = frame.resid_sums[:, None] + (deltas * sigma)[None, :] * g_sums[:, None]
        claims += randomization(design, terms, sums, sigma).claims
    return sigma, frame.n, tuple(float(c) / draws for c in claims)


def smallest_detected(
    deltas: tuple[float, ...], rates: tuple[float, ...], target: float = POWER_TARGET
) -> float | None:
    """The smallest shift above 0 detected at ``target`` or more, or ``None``."""
    for delta, rate in zip(deltas, rates, strict=True):
        if delta > 0 and rate >= target:
            return delta
    return None
