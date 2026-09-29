# Errors and refusals

This page lists every typed error tsdive raises, the check behind it, a
real message, and what to do. The command blocks ran on the demo data
when the page was built; the Python blocks run as tests.

A **refusal** is a typed error that derives from `TSDiveError`. It means
the data cannot answer the question you asked: a baseline with a clipped
stretch, archives that do not share a grid, a trial too short to test.
It is a result, and it names the check that failed. Any other exception
(`ValueError`, `TypeError`, `FileNotFoundError`) means the call itself
was wrong: a malformed window, a missing file, overlapping periods.

## Exit status

| status | meaning |
|---|---|
| 0 | the answer was printed |
| 2 | usage error or invalid input: a malformed window, a missing file, overlapping periods, an unknown flag |
| 3 | refusal: a typed error, printed as `[ErrorName] message` on stderr |

`tsdive run` keeps its own rule. It exits 0 when the ledger holds at
least one profile or finding, and a refused step is a row of that
ledger. It exits 2 when the plan cannot be read or no step produced a
result. `tsdive switchback analyze` exits 3 when its difference in means
is refused, even when the command printed the rest of its report.

A refusal on the command line:

```tsdive exit=3
tsdive screen data/demo/fic101_demo.parquet \
    --baseline 2024-03-31T00:00:00Z/2024-03-31T03:00:00Z \
    --window 2024-03-31T03:00:00Z/2024-03-31T06:00:00Z
```

Under `--json` the same refusal also prints one object on stdout, the
object the MCP server returns. The first line of the block goes to
stderr:

```tsdive exit=3
tsdive screen data/demo/fic101_demo.parquet \
    --baseline 2024-03-31T00:00:00Z/2024-03-31T03:00:00Z \
    --window 2024-03-31T03:00:00Z/2024-03-31T06:00:00Z --json
```

`result_kind` is `refusal`, `error_type` names the class, and `cause` is
the message. A script reads the exit status first and the object second.

## In Python

Catch `TSDiveError` for every refusal, and let the other exceptions
through, because they point at a bug in the calling code:

```pycon
>>> import tsdive
>>> windows = ["2024-03-30T20:00:00Z/2024-03-31T01:00:00Z",
...            "2024-03-31T00:00:00Z/2024-03-31T03:00:00Z"]
>>> for baseline in windows:
...     try:
...         s = tsdive.screen("data/demo/fic101_demo.parquet", baseline,
...                           "2024-03-31T03:00:00Z/2024-03-31T06:00:00Z")
...     except tsdive.TSDiveError as e:
...         print(type(e).__name__)
...     else:
...         print("flagged", s.to_dict()["n_flagged"])
flagged 0
InsufficientQuality

```

The message names Python keywords where the command line names flags:
`rate_s=` in Python, `--rate-s` on the command line.

## The typed errors

### TSDiveError

The base class of every refusal. `except tsdive.TSDiveError` catches all
of the classes below and nothing else. It is never raised on its own.

### SchemaError

Raised when the input breaks the archive schema: a file that is not a
tsdive archive, a missing quality column at ingest, naive timestamps
without `--tz`, a date that reads day first and month first, a metadata
key tsdive does not define, a `quality_codes` entry that names no
severity, or a string value on a tag that is not `role: MODE` and that
`quality_codes` does not name. A single-tag ingest raises it for an
export of several tags, one column naming the tag of each row.

```csv file=export.csv
timestamp,value,quality
2024-03-30T20:00:00Z,62.1,GOOD
2024-03-30T20:01:00Z,62.3,GOOD
```

```tsdive exit=3
tsdive profile export.csv
```

What to do: the message names the fix. Build an archive with
[`tsdive ingest`](cli/ingest.md), pass `--tz` or `--dayfirst`, or
correct the metadata key it names. The [archive schema](../SCHEMA.md)
lists every column and key.

### InsufficientQuality

Raised when a statistic has too few GOOD samples, and when a window used
as a baseline holds a clipped sample. A baseline needs 30 GOOD samples.

```tsdive exit=3
tsdive spc data/demo/fic101_demo.parquet \
    --baseline 2024-03-30T20:00:00Z/2024-03-30T20:20:00Z \
    --window 2024-03-30T20:20:00Z/2024-03-30T22:00:00Z
```

The screen at the top of this page shows the other cause, a clipped
baseline.

What to do: pick a baseline window that avoids the clipped stretch and
holds enough GOOD samples. `tsdive segment` shows where the stretches
are. A window that lies outside the archive's time range has no samples
at all, so check the dates against `tsdive profile` first.

### IncomparableSamplingError

Raised when two reads under different calculation bases are mixed, a
time-weighted mean with an event-weighted one. `switchback analyze`
checks the target against every covariate this way.

```pycon
>>> from datetime import datetime, timezone
>>> from tsdive import SingleFileStore
>>> from tsdive.store.sampling_contract import (
...     CalculationBasis, RetrievalMode, SamplingContract)
>>> from tsdive.store.tagstore import assert_windows_comparable
>>> store = SingleFileStore("data/demo/fic101_demo.parquet")
>>> tag = store.list_tags()[0]
>>> start = datetime(2024, 3, 30, 20, tzinfo=timezone.utc)
>>> end = datetime(2024, 3, 30, 22, tzinfo=timezone.utc)
>>> reads = [store.read_window(tag, start, end,
...              SamplingContract(basis, RetrievalMode.RECORDED))
...          for basis in CalculationBasis]
>>> assert_windows_comparable(*reads)
Traceback (most recent call last):
    ...
tsdive.errors.IncomparableSamplingError: cannot mix windows: demo:FIC101.PV
is TIME_WEIGHTED, demo:FIC101.PV is EVENT_WEIGHTED; these are different
quantities

```

What to do: read every archive under one `--basis`.

### NonMonotonicIndex

Raised when timestamps go backwards, by `ingest` before it writes the
archive and by every read of an archive. tsdive does not sort them,
because gaps over a re-sorted index would describe an order the
historian never had. The error carries the offending positions.

```pycon
>>> import pandas as pd
>>> import tsdive
>>> frame = pd.DataFrame({
...     "timestamp": pd.to_datetime(["2024-03-30 20:02", "2024-03-30 20:01",
...                                  "2024-03-30 20:03"], utc=True),
...     "value": [1.0, 2.0, 3.0], "quality": ["GOOD"] * 3})
>>> meta = tsdive.TagMeta(identity=tsdive.TagIdentity("plant1", "FI102.PV"),
...                       name="FI-102 flow")
>>> path = tsdive.write_tag("FI102.parquet", frame, meta)
>>> try:
...     tsdive.profile(path)
... except tsdive.NonMonotonicIndex as e:
...     print(e.offending_positions)
[1]

```

What to do: fix the export. A backwards step in a CSV usually comes
from dates read in the wrong order, day first or month first, or from a
DST fall-back hour written in local time without an offset.

### UnresolvedUnitError

Raised when a comparison needs a canonical unit and the tag's unit is
not in the alias table. Reports print an unknown unit as
`unresolved (null)` and carry on; only a conversion stops.

```pycon
>>> from tsdive.store.units import convert
>>> convert(1.0, "furlong/fortnight", "m/s")
Traceback (most recent call last):
    ...
tsdive.errors.UnresolvedUnitError: unit 'furlong/fortnight' is not in the
alias table; register it or declare the reference state

```

What to do: write `unit_raw` the way the [archive schema](../SCHEMA.md)
lists it, or leave it null.

### IncomparableUnitsError

Raised when two units cannot be compared: different dimensions, or a
unit at reference conditions (`Nm3/h`, `Sm3/h`, `scfm`) whose reference
state is not declared.

```pycon
>>> from tsdive.store.units import check_comparable
>>> check_comparable("m3/h", "degC")
Traceback (most recent call last):
    ...
tsdive.errors.IncomparableUnitsError: units 'm3/h' and 'degC' have
different dimensions (...)

```

What to do: compare tags of one dimension, and declare the reference
state of a normal or standard volume flow before comparing it with an
actual one.

### RegimeTooSparse

Raised by `screen --mode` when the window holds a regime the baseline
never saw, or a regime with too few GOOD baseline samples.

```tsdive
tsdive segment data/demo/fic101_demo.parquet --mode-out modes.parquet
```

```tsdive exit=3
tsdive screen data/demo/fic101_demo.parquet --mode modes.parquet \
    --baseline 2024-03-30T20:00:00Z/2024-03-31T01:00:00Z \
    --window 2024-03-31T01:00:00Z/2024-03-31T06:00:00Z
```

What to do: choose a baseline that covers every regime the window
visits, or screen without `--mode`.

### PopulationTooSparse

Raised by the population baseline of the detector studies when an
(asset, variable) pair has too few training windows, or none. No command
raises it.

```pycon
>>> from tsdive.baselines.population import match_population
>>> match_population({}, "well-7", "PDG")
Traceback (most recent call last):
    ...
tsdive.errors.PopulationTooSparse: no training window for ('well-7', 'PDG');
this pair appears only in the fold under test

```

What to do: give the pair more training windows, or report it as
refused.

### GroupLeakage

Raised when one holdout group sits on both sides of a split.
`tsdive.eval.group_holdout` checks its own result before returning it.

```pycon
>>> from tsdive.eval import GroupSplit
>>> split = GroupSplit(group_col="asset", stratum_col="fault", n_folds=2,
...                    seed=0, fold_of_group={"A": 0}, folds=(("A",), ("A",)),
...                    fold_sizes=(1, 1), strata_counts=({}, {}),
...                    group_sizes={"A": 2})
>>> split.leakage_check()
Traceback (most recent call last):
    ...
tsdive.errors.GroupLeakage: group 'A' appears in fold 0 and fold 1; a number
computed over this split would measure memorisation of 'A', not detection

```

What to do: build the split with `group_holdout`, which never divides a
group.

### MspcAlignmentError

Raised when the tags of `mspc` cannot share one UTC grid at the declared
rate with enough coverage, 0.95 by default. A collection outage in one
tag is the usual cause.

```tsdive exit=3
tsdive mspc data/demo/fic101_demo.parquet data/demo/tic101_demo.parquet \
    --baseline 2024-03-30T20:00:00Z/2024-03-31T01:00:00Z \
    --window 2024-03-31T01:00:00Z/2024-03-31T06:00:00Z
```

What to do: move the baseline off the outage, here the 40 minutes from
23:00, or lower `--min-coverage` and accept a model fitted on fewer
rows.

### DesignTooSmall

Raised when a switchback window holds too few blocks for a randomization
test at the 5% level: fewer than 20 balanced assignments, or a smallest
reachable p-value above 0.05.

```tsdive exit=3
tsdive switchback plan --window 2024-06-03T00:00:00Z/2024-06-03T04:00:00Z \
    --block PT1H --washout PT10M --seed 1 -o small.json
```

What to do: use more, shorter blocks, or a longer window. Eight blocks is
the fewest that can reach p 0.05.

### ScheduleMismatch

Raised by `switchback analyze` when the plan file was edited: its digest
does not match its blocks, a block's times do not follow from the start
and block length, the settings are not balanced, or they differ from the
ones the seed draws.

```pycon
>>> import dataclasses
>>> import tsdive
>>> from tsdive.switchback import verify_plan
>>> plan = tsdive.switchback_plan("2024-06-03T00:00:00Z", "2024-06-04T00:00:00Z",
...                               block="PT1H", washout="PT15M", seed=7)
>>> verify_plan(dataclasses.replace(plan, seed=8))
Traceback (most recent call last):
    ...
tsdive.errors.ScheduleMismatch: ...

```

What to do: restore the plan file that recorded the trial, the one whose
digest the operators ran. A plan records one randomization, so do not
re-plan a trial that already ran.

### NarratorUnavailable

Raised when narration of an evidence ledger is asked for and no LLM
endpoint is configured. tsdive has no default endpoint.

```pycon
>>> from tsdive.narrate import EvidenceLedger
>>> from tsdive.narrate.base import RemoteNarrator
>>> RemoteNarrator().narrate(EvidenceLedger(title="demo"))
Traceback (most recent call last):
    ...
tsdive.errors.NarratorUnavailable: TSDIVE_LLM_ENDPOINT is not set; there is
no default LLM endpoint. Set it explicitly to enable narration.

```

What to do: set `TSDIVE_LLM_ENDPOINT`, or read the ledger as it is.
