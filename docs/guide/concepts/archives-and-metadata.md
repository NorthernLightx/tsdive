# Archives and metadata

tsdive reads [archives](../../reference/glossary.md#archive), never raw
exports. This page covers what an archive holds, where its metadata
comes from, and which metadata keys change which checks.

## One tag per archive

An archive is one parquet file for one
[tag](../../reference/glossary.md#tag). It holds three columns and a
block of metadata:

| column | holds |
|---|---|
| `timestamp` | UTC time of the sample |
| `value` | the reading, a number, or a string state on a MODE tag |
| `quality` | the historian's quality code, kept exactly as exported |

`tsdive ingest` builds an archive from a CSV or parquet export, and
`tsdive.write_tag` builds one from a pandas DataFrame. Neither ever
edits an existing archive: a new export makes a new file.

## Identity

A tag is named by its [identity](../../reference/glossary.md#identity),
`source_id:point_id`. `source_id` names the historian or collector, such
as `plant1` or `pi-north`, and `point_id` names the tag in it, such as
`FIC101.PV`. The display name, `FIC-101 flow`, is metadata and may
change; the identity may not. Two archives with one identity are two
reads of one tag.

## Metadata keys

The metadata travels inside the archive under the `tsdive.meta` key. At
ingest it comes from a JSON file, and `ingest --init-meta` writes a
template with a comment on every key. Only `identity` and `name` are
required. Each other key turns on a check:

| key | turns on |
|---|---|
| `unit_raw` | unit resolution, printed as `units m3/h -> cubic meters per hour` |
| `eng_range_zero`, `eng_range_span` | the [clipping check](clipping-and-censoring.md); without them a report prints `censored unknown` |
| `sample_rate_s` | the `sparse_by_design` gap rule, and the grid `compare` and `mspc` align on |
| `retrieval_mode` | the second field of the [sampling contract](sampling-contract.md), `RECORDED` or `INTERPOLATED` |
| `role` | `PV`, `SP`, `OP` or `MODE`; `MODE` switches numeric statistics off |
| `quality_codes` | your site's [quality codes](quality-and-severity.md), mapped to a severity |
| `asset`, `loop_id` | grouping; no check reads them |

A key tsdive does not define is refused, so a misspelt `unit` or a
nested `eng_range` does not vanish silently:

```json file=FI2201.json
{
  "identity": {"source_id": "plant1", "point_id": "FI2201.PV"},
  "name": "FI-2201 cooling water flow",
  "unit": "m3/h"
}
```

```csv file=FI2201.csv
timestamp,value,quality
2026-03-02T07:00:00Z,41.3,GOOD
2026-03-02T07:05:00Z,41.2,GOOD
```

```tsdive exit=3
tsdive ingest FI2201.csv --out FI2201.parquet --meta FI2201.json
```

The [archive schema](../../SCHEMA.md) lists every column, key and type.

## What the metadata cannot tell

A field the source does not state stays null. tsdive never fills in a
placeholder such as a range of 0 to 100 or a rate of 1 s. A check that
needs a missing field says so in the report instead of guessing: an
undeclared engineering range prints `censored unknown`, and `compare`
without a sample rate refuses its pair and joint tables.

## What to do

- Run `tsdive ingest <export> --init-meta META.json` and fill in the
  template before the first ingest.
- Declare the engineering range and the sample rate. They cost two
  lines and turn on the checks that catch a pegged transmitter and a
  missing hour.
