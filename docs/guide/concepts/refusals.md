# Refusals

When the data you gave cannot answer a question, tsdive raises a typed
error instead of printing a number. That error is a
[refusal](../../reference/glossary.md#refusal). This page covers what a
refusal looks like, which number it replaces, and what to do with one.

## A check with no answer raises

Each check needs something from the data. A baseline needs 30 GOOD
samples and no clipped one. `mspc` needs a grid with 0.95 of its cells
filled. A switchback trial needs at least eight blocks. When the data
cannot give it, the command stops and names the check:

```tsdive exit=3
tsdive mspc data/demo/fic101_demo.parquet data/demo/tic101_demo.parquet \
    --baseline 2024-03-30T20:00:00Z/2024-03-31T01:00:00Z \
    --window 2024-03-31T01:00:00Z/2024-03-31T06:00:00Z
```

The line has three parts: the error class in brackets,
`MspcAlignmentError`; what the data showed, aligned coverage 0.867; and
the rule it broke, 0.95 required. The exit status is 3.

## Why not a number

Each refusal replaces a number that would look fine and mean nothing:

- limits from a censored baseline would call the real excursion normal;
- a model fitted across a 40-minute hole would have to invent 40 minutes
  of flow;
- a p-value from four blocks cannot reach 0.05 whatever the process did.

A refusal is a result: it tells you what to fix in the question.

## Refusals are results in every interface

- On the command line: `[ErrorName] message` on stderr and exit status
  3. Under `--json`, stdout also carries one object with `result_kind`
  `refusal`, `error_type` and `cause`.
- In Python: an exception that derives from `tsdive.TSDiveError`.
- In a `tsdive run` ledger: a row of the `refusals` list, beside the
  findings. The run still exits 0 when anything else was found. Under
  `--strict` it exits 3, or 2 when a step also raised a usage error.
- In `compare`: a tag whose read is refused keeps its row, with the
  error class in the `quality` column.
- Through the MCP server: a result with `result_kind` `refusal`, not a
  tool error.

A usage error is different: a malformed window, a missing file or an
unknown flag exits 2, and in Python it raises `ValueError`,
`TypeError` or `FileNotFoundError`. That points at the call, not the
data. In a `tsdive run` ledger it is a row of the `errors` list.

## What to do

Read the message: it names the check and usually the fix. The
[errors and refusals](../../reference/errors.md) page lists every class
with a real message and the usual remedy. The common ones:

| refusal | usual fix |
|---|---|
| `InsufficientQuality`, censored baseline | move the baseline off the clipped stretch |
| `InsufficientQuality`, too few GOOD samples | widen the baseline, or check it lies inside the archive's time range |
| `SchemaError` at ingest | pass `--tz`, `--dayfirst`, or fix the metadata key it names |
| `MspcAlignmentError` | move the baseline off the outage, or declare `sample_rate_s` |
| `DesignTooSmall` | use more, shorter switchback blocks |
