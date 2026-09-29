# Ingest a PI, IP.21 or OPC export

This guide turns the CSV files a historian exports into archives. It
covers local time, day-first dates, digital states in the value column,
quality codes, interpolated exports and wide files. Each step runs the
real command on a small export.

## The recipe

1. Write a metadata template with `tsdive ingest <export> --init-meta META.json`.
2. Fill in the template: identity, name, unit, engineering range,
   sample rate, quality codes.
3. Ingest with the flags your export needs: `--tz`, `--dayfirst` or
   `--timestamp-format`, and the column names.
4. Profile the archive and read the headline.

## A PI export in local time

A PI DataLink export often has local timestamps in the regional date
order, and writes digital states such as `I/O Timeout` into the value
column. It has no quality column:

```csv file=TI3101.csv
Timestamp,TI3101.PV
28/10/2026 06:00:00,181.2
28/10/2026 06:05:00,181.4
28/10/2026 06:10:00,I/O Timeout
28/10/2026 06:15:00,I/O Timeout
28/10/2026 06:20:00,181.9
28/10/2026 06:25:00,182.1
```

Write the template, naming the columns the export uses:

```tsdive
tsdive ingest TI3101.csv --init-meta TI3101.json \
    --timestamp-col Timestamp --value-col TI3101.PV
```

The `columns` comment in the template says what ingest found: no
quality column. Fill in the tag, and name the digital state in
`quality_codes` so ingest reads it as a BAD sample instead of refusing a
string in a numeric column:

```json file=TI3101.json
{
  "identity": {"source_id": "pi-north", "point_id": "TI3101.PV"},
  "name": "TI-3101 reactor outlet temperature",
  "unit_raw": "degC",
  "eng_range_zero": 0.0,
  "eng_range_span": 250.0,
  "sample_rate_s": 300,
  "quality_codes": {"I/O Timeout": "BAD"}
}
```

With no quality column, `--assume-quality GOOD` states the quality of
every other sample. The archive records that the quality was assumed,
and every profile of it says so. `--tz` and `--dayfirst` place the local
dates in UTC:

```tsdive
tsdive ingest TI3101.csv --out TI3101.parquet --meta TI3101.json \
    --timestamp-col Timestamp --value-col TI3101.PV \
    --tz Europe/Berlin --dayfirst --assume-quality GOOD
```

```tsdive lines=16
tsdive profile TI3101.parquet
```

`GOOD 4/6` and `BAD 2`: the two `I/O Timeout` rows are BAD and their
values are nulled, so they stay out of the statistics. 06:00 in Berlin on
28 October 2026 is 05:00 UTC, because the clocks went back on 25 October.

## An IP.21 export with its own date format

IP.21 and many SCADA systems write dates such as `28-Oct-2026 06:00:00`.
`--timestamp-format` takes a
[strptime format](https://docs.python.org/3/library/datetime.html#format-codes)
for them:

```csv file=PI4402.csv
TS,VALUE,STATUS
28-Oct-2026 06:00:00,3.52,Good
28-Oct-2026 06:01:00,3.55,Good
28-Oct-2026 06:02:00,3.51,Good
```

This export holds values the historian interpolated every minute, not
the values it stored. Say so with `retrieval_mode`, so every report of
the archive states it:

```json file=PI4402.json
{
  "identity": {"source_id": "ip21", "point_id": "PI4402.PV"},
  "name": "PI-4402 reactor pressure",
  "unit_raw": "bar",
  "sample_rate_s": 60,
  "retrieval_mode": "INTERPOLATED"
}
```

```tsdive
tsdive ingest PI4402.csv --out PI4402.parquet --meta PI4402.json \
    --timestamp-col TS --value-col VALUE --quality-col STATUS \
    --tz Europe/Berlin --timestamp-format "%d-%b-%Y %H:%M:%S"
```

```tsdive lines=5
tsdive profile PI4402.parquet
```

The contract line reads `INTERPOLATED`, and the digest changed with it.

## An OPC export with byte quality codes

Classic OPC DA writes quality as a byte: 192 Good, 64 Uncertain, 0 Bad,
with sub-status values around them. Read as OPC UA codes they mean
something else, so declare them:

```csv file=FIC501.csv
SourceTimestamp,Value,StatusCode
2026-10-28T05:00:00Z,12.40,192
2026-10-28T05:01:00Z,12.38,192
2026-10-28T05:02:00Z,12.41,216
2026-10-28T05:03:00Z,0.00,0
2026-10-28T05:04:00Z,12.39,192
2026-10-28T05:05:00Z,12.44,64
```

```tsdive
tsdive ingest FIC501.csv --init-meta FIC501.json \
    --timestamp-col SourceTimestamp --value-col Value --quality-col StatusCode
```

The template lists every code the file holds, each waiting for a
severity. `216` is a Good sub-status (local override) in OPC DA:

```json file=FIC501.json
{
  "identity": {"source_id": "opc-line5", "point_id": "FIC501.PV"},
  "name": "FIC-501 feed flow",
  "unit_raw": "m3/h",
  "sample_rate_s": 60,
  "quality_codes": {"192": "GOOD", "216": "GOOD", "64": "UNCERTAIN", "0": "BAD"}
}
```

The timestamps carry `Z`, so no `--tz` is needed:

```tsdive
tsdive ingest FIC501.csv --out FIC501.parquet --meta FIC501.json \
    --timestamp-col SourceTimestamp --value-col Value --quality-col StatusCode
```

```tsdive lines=14
tsdive profile FIC501.parquet
```

The BAD sample at 05:03 carries a value of 0.00 that is not a flow; it
stays out of every statistic.

## A wide export

A wide export has one column per tag, and its quality columns share a
suffix such as `_q`. Two commands handle it, as the
[usage walkthrough](../../usage.md#ingest) shows: `--wide --init-meta
DIR` writes one template per tag, and `--wide --out DIR --meta-dir DIR`
writes one archive per tag.

## When ingest refuses

| message says | do |
|---|---|
| `is naive (no UTC offset)` | pass `--tz` with the zone the export was written in |
| `reads as day 02 of month 03 or as month 02` | pass `--dayfirst`, or `--timestamp-format` |
| `cannot localise ... DST transition` | export the stretch around the clock change with UTC offsets |
| `no quality column` | name it with `--quality-col`, or pass `--assume-quality` |
| `unknown key` | fix the metadata key it names; the message suggests the right one |
| `is not numeric and no digital state explains it` | name the string in `quality_codes` |
| `precedes the row before it` | fix the row order in the export; tsdive does not sort it |

The [errors page](../../reference/errors.md#schemaerror) lists every
schema refusal.
