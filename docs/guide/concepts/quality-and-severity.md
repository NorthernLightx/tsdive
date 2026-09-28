# Quality codes and severity

A historian stores a [quality code](../../reference/glossary.md#quality-code)
beside every value. tsdive keeps the code exactly as exported and
derives a [severity](../../reference/glossary.md#severity) from it:
GOOD, UNCERTAIN or BAD. Only GOOD samples with a finite value feed the
statistics.

## How a code becomes a severity

The first rule that matches wins:

1. The tag's own `quality_codes` map in its metadata.
2. The numeric digital states tsdive ships: 248 Bad, 249 Comms Outage,
   250 Scan Off, 251 Substituted, 257 Over Range. The value on such a row
   is nulled, so a state number such as 257 never enters a statistic as
   257.0 m3/h.
3. The string codes `GOOD`, `UNCERTAIN`, `SUBSTITUTED`, `SCAN OFF`, `BAD`,
   `OVER RANGE`, `UNDER RANGE` and `COMM FAILURE`, in any case.
4. OPC UA status codes. The top bits mark UNCERTAIN or BAD, and GOOD
   needs one of the Good codes tsdive ships.
5. Anything else is UNCERTAIN and listed as unmapped.

A code tsdive cannot read never counts as GOOD. The demo flow has one
sample with the code `SENSOR DRIFT`, which no rule knows:

```tsdive lines=16
tsdive profile data/demo/fic101_demo.parquet
```

`GOOD 561/562` in the headline and `GOOD 561   UNCERTAIN 1   BAD 0` in
the Quality section count the severities. `unmapped codes, treated
UNCERTAIN: SENSOR DRIFT` names the code to declare.

## Coverage and valid are different

`coverage` asks whether the historian stored a row at all. `valid` asks
whether the row was usable: GOOD and a finite number. The demo flow has
coverage 0.933, because 40 minutes hold no row, and valid 0.998, because
561 of its 562 rows are usable. A historian that writes a row every scan
whatever happens shows the difference: coverage stays at 1.000 while
valid falls.

## Declaring your site's codes

Map every code the report lists as unmapped in the tag's metadata:

```json
"quality_codes": {"SENSOR DRIFT": "UNCERTAIN", "Questionable": "UNCERTAIN", "192": "GOOD"}
```

The map is consulted before every shipped table, and matching ignores
case and surrounding spaces. A code mapped to a severity that does not
exist raises `SchemaError`.

Two cases need the map:

- OPC DA byte qualities, `192` Good, `64` Uncertain, `0` Bad. They read
  as OPC UA codes otherwise, and `0` would mean Good there. When every
  unmapped code is a byte-sized integer, the report says so.
- Digital states in the value column, such as PI's `I/O Timeout`. Name
  the string in `quality_codes` and ingest nulls the value at that
  severity instead of refusing the file.

## What to do

- Read the `unmapped codes` line of every first profile and declare each
  code it lists.
- Treat a fall in `valid` with steady `coverage` as a sensor or
  transmitter problem, not a collection problem.
