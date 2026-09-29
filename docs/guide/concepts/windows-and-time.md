# Windows and time zones

Every command reads a [window](../../reference/glossary.md#window), a
span of time with a start and an end. This page covers the four ways to
write one, why tsdive stores UTC only, and what to do with local
timestamps.

## Four ways to write a window

A window is ISO 8601 text. Both ends are included.

| form | example | reads |
|---|---|---|
| `START/END` | `2024-03-30T20:00:00Z/2024-03-31T01:00:00Z` | 20:00 to 01:00 UTC |
| `START/DURATION` | `2024-03-30T20:00:00Z/PT5H` | the same five hours |
| `DURATION/END` | `PT5H/2024-03-31T01:00:00Z` | the same five hours |
| a date | `2024-03-31` | the whole UTC day, 00:00 to 00:00 the next day |

A duration counts days, hours, minutes and seconds: `PT5H`, `P2DT6H`,
`PT90M`. Months and years are refused, because their length varies. The
same five hours, written two ways:

```tsdive lines=4
tsdive profile data/demo/fic101_demo.parquet --window 2024-03-30T20:00:00Z/PT5H
tsdive profile data/demo/fic101_demo.parquet --window PT5H/2024-03-31T01:00:00Z
```

## Every bound carries its offset

A bound needs `Z` or an offset such as `+02:00`. tsdive converts an
offset to UTC and prints UTC. A bound without one is refused, because
tsdive cannot tell which instant it names:

```tsdive exit=2
tsdive profile data/demo/fic101_demo.parquet --window 2024-03-30T20:00:00/PT5H
```

The same window with a Berlin offset reads 19:00 to 00:00 UTC:

```tsdive lines=4
tsdive profile data/demo/fic101_demo.parquet --window 2024-03-30T20:00:00+01:00/PT5H
```

## Local time in exports

Historian exports often write local time with no offset, such as
`02/03/2026 08:00`. That is a
[naive timestamp](../../reference/glossary.md#naive-timestamp).
`tsdive ingest --tz Europe/Berlin` states the zone it was written in,
and ingest stores UTC. tsdive never assumes UTC for you.

Two more facts decide how an export's dates read:

- A date such as `02/03/2026` is 2 March day first and 3 February month
  first. Ingest refuses it until you pass `--dayfirst`, or
  `--timestamp-format` with a
  [strptime format](https://docs.python.org/3/library/datetime.html#format-codes)
  such as `%d/%m/%Y %H:%M`.
- In autumn a local hour repeats when the clocks go back. Ingest places
  its first pass at the summer offset and its second at the winter one,
  by row order. An export that holds the hour once, or out of time
  order, raises `SchemaError`.
- In spring one local hour does not exist, and a time in it raises
  `SchemaError`. Export the stretch around either change with UTC
  offsets.

## DST inside a window

`profile --tz` names zones whose clock changes inside the window the
report should list. The demo window holds the change in Europe/London at
01:00 UTC on 31 March 2024. It changes nothing in the data, because the
archive is in UTC. It tells you that a shift report in local time has an
hour less that night:

```tsdive lines=24
tsdive profile data/demo/fic101_demo.parquet --tz Europe/London
```

## What to do

- Write windows with `Z`, or with the offset of the clock you read them
  from.
- Ingest local exports with `--tz`, and with `--dayfirst` when the
  dates are day first.
- Convert the UTC times a report prints before you look them up on a
  local trend.
