# Use tsdive from Python and pandas

Every command is a function that returns an object: its `render()` is
the text the command prints, its `to_dict()` the JSON `--json` prints,
and its `.frame` a pandas DataFrame. This guide builds an archive from a
DataFrame, runs the analyses, and handles refusals. Every snippet runs
as a test against the demo data that `tsdive demo data` writes.

## Get the demo data

```pycon
>>> import tsdive
>>> paths = tsdive.write_demo_data("data-copy")
>>> [p.name for p in paths[:2]]
['fic101_demo.parquet', 'tic101_demo.parquet']

```

## An archive from a DataFrame

`write_tag` takes a DataFrame with `timestamp` (UTC-aware), `value` and
`quality`, and the tag's metadata. Declare `sample_rate_s`: `compare`
and `mspc` align tags on it, and a hole of up to 3 times it counts as
normal spacing instead of data loss.

```pycon
>>> import numpy as np
>>> import pandas as pd
>>> stamps = pd.date_range("2024-03-30 20:00", periods=120, freq="min", tz="UTC")
>>> frame = pd.DataFrame({
...     "timestamp": stamps,
...     "value": 41.0 + np.sin(np.arange(120) / 10),
...     "quality": "GOOD",
... })
>>> meta = tsdive.TagMeta(
...     identity=tsdive.TagIdentity("plant1", "FI2201.PV"),
...     name="FI-2201 cooling water flow",
...     unit_raw="m3/h",
...     eng_range=tsdive.EngRange(zero=0.0, span=80.0),
...     sample_rate_s=60.0,
... )
>>> path = tsdive.write_tag("FI2201.parquet", frame, meta)
>>> p = tsdive.profile(path)
>>> round(p.physics.coverage.coverage, 3), p.physics.clipping.censored_verdict
(1.0, False)

```

An existing file raises `FileExistsError`: an archive is written once.
Pass `overwrite=True` to replace the whole file.

## Results into pandas

Each result has a `.frame` whose rows depend on the analysis:

| result | `.frame` holds |
|---|---|
| `Profile` | every sample of the window: `timestamp`, `value`, `quality`, `severity`, `valid` |
| `SegmentAnalysis` | one row per segment: `index`, `start`, `end`, `n`, `median`, `mad` |
| `ScreenAnalysis` | the flagged samples only |
| `SpcAnalysis` | one row per rule hit: `timestamp`, `rule`, `detail` |
| `MspcAnalysis` | one row per aligned timestamp: `timestamp`, `t2`, `spe`, `t2_breach`, `spe_breach` |
| `CompareAnalysis` | one row per tag, the tag table |

```pycon
>>> m = tsdive.mspc(["data/demo/fic101_demo.parquet", "data/demo/tic101_demo.parquet"],
...                 "2024-03-30T20:00:00Z/2024-03-30T23:00:00Z",
...                 "2024-03-31T04:00:00Z/2024-03-31T06:00:00Z")
>>> m.frame.columns.tolist()
['timestamp', 't2', 'spe', 't2_breach', 'spe_breach']
>>> int(m.frame["spe_breach"].sum()), len(m.frame)
(108, 121)
>>> hourly = m.frame.set_index("timestamp")["spe_breach"].resample("h").sum()
>>> hourly.astype(int).tolist()
[53, 54, 1]

```

`to_dict()` returns the document the command prints under `--json`,
ready for `json.dumps`. It holds everything the text report shows:

```pycon
>>> s = tsdive.screen("data/demo/fic101_demo.parquet",
...                   "2024-03-30T20:00:00Z/2024-03-31T01:00:00Z",
...                   "2024-03-31T01:00:00Z/2024-03-31T06:00:00Z")
>>> doc = s.to_dict()
>>> doc["n_flagged"], doc["n_screened"], doc["method"]
(29, 300, 'MAD')
>>> report = tsdive.profile("data/demo/fic101_demo.parquet").to_dict()
>>> sorted(report)
['contract', 'coverage', 'flatline', 'name', 'quality', 'range', 'tag', 'timestamps',
 'units', 'values', 'window']
>>> report["range"]["n_clipped"], report["coverage"]["n_data_loss_gaps"]
(29, 1)

```

## Refusals and errors

A refusal raises a subclass of `tsdive.TSDiveError`. Any other
exception means the call was wrong. Catch the first and let the second
through:

```pycon
>>> for baseline in ["2024-03-30T20:00:00Z/2024-03-31T01:00:00Z",
...                  "2024-03-31T00:00:00Z/2024-03-31T03:00:00Z",
...                  "2024-03-31T01:00:00Z/2024-03-31T02:00:00Z"]:
...     try:
...         _ = tsdive.spc("data/demo/fic101_demo.parquet", baseline,
...                        "2024-03-31T03:00:00Z/2024-03-31T06:00:00Z")
...         print("ok")
...     except tsdive.TSDiveError as e:
...         print(type(e).__name__)
ok
InsufficientQuality
ok

```

The messages name Python keywords, such as `rate_s=`, where the
command line names flags. [Errors and refusals](../../reference/errors.md)
lists every class.

## Plotting

tsdive draws nothing on screen. `.frame` feeds any plotting library:

```python
ax = p.frame.plot(x="timestamp", y="value")
```

The [API reference](../../reference/api/index.md) lists every function
and class with its parameters and an example.
