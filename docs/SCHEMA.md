# Archive schema

What a parquet file has to contain before tsdive will read it, and how
to write one from your own export; per-source notes are in SOURCES.md.

## File layout

A `TagStore` is a directory of single-tag parquet files:

```
root/
  <source_id>/
    <point_id>.parquet
    <point_id>.parquet
  <source_id>/
    <point_id>.parquet
```

`source_id` is the historian or collector the tag came from; `point_id` is
its identifier inside that historian. Together they are the tag identity,
which is what every read, refusal and provenance record keys on. The
display name is metadata, not identity, because control engineers rename
tags and identity has to survive that.

`SingleFileStore` reads one explicit path instead and makes no assumption
about the surrounding directory, which is what the CLI uses.

## Required columns

| column | dtype | rule |
|---|---|---|
| `timestamp` | `datetime64[ns, UTC]` (any tz-aware datetime64) | must be timezone-aware |
| `value` | numeric, or null (string only on `role: MODE` tags) | digital states are nulled on read, never coerced |
| `quality` | string or integer | mandatory; stored verbatim |

Extra columns are carried through untouched.

A `value` column without a `quality` column raises `SchemaError`. There is
no default quality: a number whose trustworthiness is unknown is not a
measurement, and tsdive will not guess one.

## Timezone rule

Timestamps are stored in UTC and only UTC. Naive timestamps are rejected
at ingest with `SchemaError`, and the CLI rejects naive `--window` bounds
the same way. Local time is a display concern; convert on the way in.

Local-time effects are reported rather than silently applied. Pass
`--tz Europe/London` (or `tz_names=` to `read_window`) and the timestamp
audit names any DST transition that falls inside the window, alongside
duplicate and non-monotonic timestamp counts.

## Quality column semantics

The raw code is preserved exactly as written. Beside it, `read_window`
derives a `severity` of `GOOD`, `UNCERTAIN` or `BAD`, and a boolean
`valid` column that combines severity with a finite value check. Severity
is derived, never substituted for the raw code.

Precedence when deriving severity:

0. **The tag's own `quality_codes` map**, when its metadata declares one
   (see below). A code the source's owner has explained beats every
   shipped table, and is not reported as unmapped.
1. **Numeric digital states.** These are archive state codes, not
   measurements. The table ships with 248 Bad, 249 Comms Outage, 250 Scan
   Off, 251 Substituted, 257 Over Range; register your site's own set in
   `store/quality.py`. The `value` on such a row is forced to null, so a
   state number like 257 can never appear in statistics as 257.0
   engineering units.
2. **String codes.** `GOOD`, `UNCERTAIN`, `SUBSTITUTED`, `SCAN OFF`,
   `BAD`, `OVER RANGE`, `UNDER RANGE`, `COMM FAILURE`. Matching is
   case-insensitive and whitespace-trimmed.
3. **OPC UA status codes.** A numeric code that reaches this rule is
   read as an OPC UA 32-bit StatusCode. The rule is deliberately
   asymmetric. Top bits `01` (Uncertain) or `10`/`11` (Bad) are enough to
   mark a sample UNCERTAIN or BAD. They are not enough to mark one GOOD.
   `GOOD` requires
   the code to match a registered Good status code in `OPC_UA_GOOD_CODES`
   (`store/quality.py`): `0x00000000` Good, plus the documented
   `0x00SS0000` Good subcodes such as `GoodNoData` and `GoodEdited`.
   Bits 15-0 are info bits, which qualify a status without changing it,
   so they are stripped before the lookup; when InfoType (bits 11-10) is
   `00` the spec requires the rest of them to be zero, so a value like
   `1` or `192` is not a *UA* status code.
4. **Anything else** maps to `UNCERTAIN`, and the code is listed in the
   report under "unmapped quality codes". Unmapped never maps to GOOD,
   including an integer whose top two bits read `00` that no registered
   Good code matches. Extend `OPC_UA_GOOD_CODES` with your site's codes
   rather than widening the rule.

### OPC DA byte qualities are a different code space

Classic OPC DA (and the PI, Kepware and similar CSV exports that carry
its qualities through) writes quality as an 8-bit byte, not a UA
StatusCode: `192` (0xC0) is Good, `64` (0x40) Uncertain, `0` Bad, with
sub-status variants spread through `0xC0-0xFF` Good, `0x40-0xBF`
Uncertain and `0x00-0x3F` Bad. The two spaces disagree on both ends:
`0` means Good in UA and Bad in DA. tsdive does not infer which space
an export writes in.

Rule 3 above is the UA reading, and it is fail-safe in only one
direction: a DA `192` lands in UNCERTAIN and is reported unmapped, but a
DA `0` would read as UA Good, which is exactly backwards. **A DA-quality
export must declare its codes in the tag's `quality_codes`**, which is
consulted before every shipped table:

```json
{"192": "GOOD", "64": "UNCERTAIN", "0": "BAD"}
```

Add the sub-status variants your historian actually emits (`193`, `216`,
`28`, …) as further keys. When the report shows unmapped codes that are
all byte-sized integers and the tag declares no `quality_codes`, it says
so on that line, which points at an OPC DA export.

Severity has a rank order (`BAD` < `UNCERTAIN` < `GOOD`) used by
`min_severity`. Do not compare `Severity` members with `<`/`>=` directly;
it is a `StrEnum`, so those operators compare alphabetically.

## Value dtype and MODE tags

After digital-state rows are nulled, what is left in `value` must be
numeric. A leftover string on a measurement tag raises `SchemaError`
naming the tag and the first offending value: nulling it would delete
data nobody said was unusable, and coercing it would invent a number.

The one legal string-valued path is a tag whose metadata declares
`role: MODE`. A mode tag's value *is* a state label (`"R1"`), so it
survives the read verbatim, its rows count as valid, and no numeric
physics is computed from it. `extract` refuses the mean and the spread
features with the note `mean refused: state-valued tag (role=MODE)`
rather than averaging names. Dwell time and change counts are still
reported, because a state change is a change in the same sense a
measurement's is.

## Units

`unit_raw` is the historian's own spelling and is stored verbatim.
Resolution to a canonical name goes through one explicit table,
`src/tsdive/store/data/engunits_aliases.csv`, matched exactly first
and then case-folded. A spelling the table does not carry resolves to
`unit_canonical: null`, the report prints `unresolved (null); comparison
refused`, and any cross-tag comparison raises `UnresolvedUnitError`. No
spelling is ever guessed into a family.

Adding an alias is one row: `alias_raw,canonical,pint_units,
reference_condition`. `pint_units` must be a string pint can parse. A
test walks every row and fails on one that is not.

Three kinds of spelling are deliberately left out of the table:

- **Ambiguous symbols.** A bare `C` is coulombs as readily as Celsius,
  and no rule in the table can tell which one a given historian means.
  `degC`, `oC` and `°C` are registered; `C` is not, and a tag spelled
  that way stays unresolved until the site says which it is.
- **Gauge pressures.** `barg` and `psig` are read against local
  atmosphere; `bar` and `psi` are absolute. The table has no column for
  a pressure datum, so registering `barg` as `bar` would silently make
  every comparison wrong by about one atmosphere. They stay unresolved
  rather than resolve to a lie. This is the same refusal
  `reference_condition` makes for `Nm3/h` against `m3/h`, which *does*
  have a column, because a reference state is declarable per comparison
  (`check_comparable(..., reference_states_declared=True)`) and a
  pressure datum, as the schema stands, is not.
- **Reference-condition flow without a declared state.** `Nm3/h`,
  `Sm3/h` and `scfm` resolve and carry `reference_condition: true`.
  Comparing one against plain `m3/h` raises `IncomparableUnitsError`
  until the caller declares the reference state.

## Embedded metadata: `tsdive.meta`

Tag metadata travels inside the parquet file, in the schema's key-value
metadata under the key `tsdive.meta`, as a JSON object. A file without
it is refused: tsdive will not invent an identity for an archive.

| field | type | meaning |
|---|---|---|
| `identity.source_id` | string | historian or collector id (required) |
| `identity.point_id` | string | tag id within that source (required) |
| `name` | string | display name (required, may be renamed freely) |
| `unit_raw` | string or null | unit exactly as the historian states it |
| `unit_canonical` | string or null | resolved unit; left null when unresolved |
| `eng_range_zero` | number or null | engineering range zero |
| `eng_range_span` | number or null | engineering range span; needed for clipping detection |
| `sample_rate_s` | number or null | declared scan rate, used to classify sparse gaps |
| `asset` | string or null | unit or equipment the tag belongs to |
| `loop_id` | string or null | control loop id |
| `role` | `PV`/`SP`/`OP`/`MODE` or null | role within that loop |
| `quality_codes` | object or null | this source's raw quality codes mapped to `GOOD`/`UNCERTAIN`/`BAD` |
| `quality_assumed` | bool | true when the quality column was written at ingest, not by the source |

Every nullable field means what it says: unknown is `None`, not a
placeholder like `0` or `1.0`. `eng_range` and `sample_rate_s` in
particular change what the physics layer can assert - without an
engineering range there is no clipping verdict, and without a declared
sample rate a wide gap cannot be classified `sparse_by_design`.

### Per-source quality codes

Historians emit codes no general table can know. Declare them on the tag
instead of editing the library:

```python
meta = TagMeta(
    identity=TagIdentity(source_id="plant1", point_id="FIC101.PV"),
    name="FIC-101 flow",
    quality_codes={"SUB": "UNCERTAIN", "OK": "GOOD"},
)
```

Keys are the raw codes exactly as the archive writes them (matched
trimmed and case-insensitively as a fallback); values must be `GOOD`,
`UNCERTAIN` or `BAD`, and any other value raises `SchemaError`. The map
is consulted before the digital-state, string and OPC UA tables, travels
inside `tsdive.meta` with the archive, and a code it covers stops
being reported as unmapped, because the site has stated what it means.

## Writing an archive

`write_tag` is the only write path. It validates the schema, embeds the
metadata, and creates the file. It does not edit archives: an existing
path raises `FileExistsError` unless you pass `overwrite=True`, which
replaces the whole file rather than patching part of it.

```python
import pandas as pd

from tsdive.store import EngRange, Role, TagIdentity, TagMeta, write_tag

frame = pd.DataFrame(
    {
        "timestamp": pd.to_datetime(["2024-03-01T00:00:00Z", "2024-03-01T00:00:03Z"], utc=True),
        "value": [62.1, 62.4],
        "quality": ["GOOD", "SENSOR DRIFT"],  # verbatim; unmapped codes become UNCERTAIN
    }
)
meta = TagMeta(
    identity=TagIdentity(source_id="plant1", point_id="FIC101.PV"),
    name="FIC-101 flow",
    unit_raw="m3/h",
    eng_range=EngRange(zero=0.0, span=100.0),
    sample_rate_s=3.0,
    role=Role.PV,
)
write_tag("archive/plant1/FIC101.PV.parquet", frame, meta)
```

Then profile it:

```console
tsdive profile archive/plant1/FIC101.PV.parquet \
    --window "2024-03-01T00:00:00Z/2024-03-01T01:00:00Z"
```

## Ingesting an export

`tsdive ingest` builds an archive from a CSV or parquet export:

```console
tsdive ingest export.csv \
    --out archive/plant1/FIC101.PV.parquet --meta meta.json \
    --timestamp-col ts --value-col v --quality-col q [--tz Europe/London]
```

`--meta` is a JSON file holding exactly the `tsdive.meta` object
documented above, so the metadata you write for ingest is the metadata
you read back off the archive. Only the three named columns are carried
over; anything else in the export is left behind, so the archive holds
exactly what its schema promises.

Timestamps that already carry an offset are converted to UTC and `--tz`
is ignored. Naive timestamps need `--tz <IANA zone>`: they are localised
to it and then converted. Without `--tz` the ingest raises `SchemaError`,
because an export with no offset does not name an instant. Local times that a DST transition makes ambiguous or
nonexistent are refused too, naming the zone.

A missing quality column is refused for the same reason a missing
quality column is refused everywhere else. `--assume-quality
GOOD|UNCERTAIN|BAD` is the explicit override; it writes that code on
every row, sets `quality_assumed: true` in the archive's metadata, and
every profile of the archive then prints a `quality ASSUMED at ingest`
line. It is refused outright when the source does have a quality column:
the source's own codes are never overwritten.

## Refusals you should expect

| condition | error |
|---|---|
| missing `timestamp`, `value` or `quality` | `SchemaError` |
| naive timestamps | `SchemaError` |
| non-numeric `value` on a tag that is not `role: MODE` | `SchemaError`, naming the tag and value |
| no `tsdive.meta` on the file | `SchemaError` |
| `tsdive.meta` missing a required key | `SchemaError`, naming the key |
| `quality_codes` naming a severity that does not exist | `SchemaError` |
| ingesting naive timestamps without `--tz` | `SchemaError` |
| ingesting a source with no quality column and no `--assume-quality` | `SchemaError` |
| `--assume-quality` on a source that has a quality column | `SchemaError` |
| writing over an existing archive | `FileExistsError` |
| comparing windows built on different sampling contracts | `IncomparableSamplingError` |
| comparing a reference-condition unit with its plain cousin | `IncomparableUnitsError` |
| statistics over a window with no GOOD samples | `InsufficientQuality` |
