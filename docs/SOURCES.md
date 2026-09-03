# Sources

tsdive profiles process time series from any source. A source is
anything that answers a window read under a stated sampling contract and
returns the samples with their physics block. The `Source` protocol in
`tsdive.store.source` is that contract, and the parquet stores are its
first implementation. A historian is one kind of source among several.

## Adapter matrix

| Source | How it gets in today | Sampling contract | Quality codes | Compression / deadband | Clock | Status |
|---|---|---|---|---|---|---|
| tsdive parquet archive | native; `write_tag` | declared on every read | verbatim + `quality_codes` | gap classes | UTC required | shipped |
| CSV / parquet export | `tsdive ingest`, one tag per file or `--wide` | declared on every read | source column, or `--assume-quality` recorded | gap classes | `--tz` required for naive stamps | shipped |
| OSIsoft PI export | `tsdive ingest` + `quality_codes` | caller declares AF calculation basis and retrieval mode | PI string codes; `quality_codes` for site codes | compression silence is a gap class | as exported | via ingest |
| OPC UA history | `tsdive ingest` on a history dump | caller declares Average or TimeAverage, and stepped-ness | StatusCode severity bits + Good whitelist | deadband holds are gaps | as exported | via ingest; client planned |
| InfluxDB / TimescaleDB export | `tsdive ingest` | retention downsampling is an aggregate contract; declare it | none in source; `--assume-quality` | continuous-query holes are gaps | as exported | via ingest |
| MQTT / Kafka capture | not yet | event-weighted by nature | none; `--assume-quality` | publisher deadband is compression | broker and device clocks undeclared | planned |
| 3W dataset | `scripts/convert_3w.py` | recorded, 1 s | none; `GOOD_ASSUMED` / `NO_DATA` | none observed | naive index localised as UTC, recorded | shipped |
| SKAB dataset | `examples/studies/skab/build_archives.py` | recorded, 1 s | none; `GOOD` assumed, `quality_assumed` recorded | none observed | naive stamps localised as UTC, stated at ingest | shipped |
| TEP simulation | `scripts/convert_tep.py` | recorded, 180 s | none; `SIMULATED` | none | synthetic clock, recorded | shipped |

## What every source must declare

- Time zone of its timestamps, or UTC-aware stamps. Naive stamps are
  refused unless the zone is stated.
- The sampling contract each read is made under (calculation basis,
  retrieval mode, aggregate type, stepped-ness).
- Its quality vocabulary through `quality_codes`, or `quality_assumed`
  when it has none. A source with neither raises `SchemaError` at ingest.
- Engineering range, or none. Without one no clipping verdict is issued.

## Rule for a new source

A source ships when it has a converter or a `Source` implementation, a
benchmark row, and a test. What it cannot declare from the list above is
recorded as a refusal or an assumption on the archive, never defaulted.
