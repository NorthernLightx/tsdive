# The sampling contract

Every read states how its numbers were made. That statement is the
[sampling contract](../../reference/glossary.md#sampling-contract), and
the profile prints it on the `contract` line:

```tsdive lines=5
tsdive profile data/demo/fic101_demo.parquet
```

`TIME_WEIGHTED  RECORDED  NONE  stepped no  digest d59d433d9c62` is four
fields and a digest.

## The four fields

| field | value here | means | change it with |
|---|---|---|---|
| [calculation basis](../../reference/glossary.md#calculation-basis) | `TIME_WEIGHTED` | a sample counts for the time until the next sample, so a value that held for an hour weighs sixty times one that held for a minute | `--basis EVENT_WEIGHTED` for counts and events, where every sample counts once |
| [retrieval mode](../../reference/glossary.md#retrieval-mode) | `RECORDED` | the values are the ones the historian stored | `retrieval_mode: INTERPOLATED` in the metadata, for an export of values the historian computed at fixed times |
| [aggregate type](../../reference/glossary.md#aggregate-type) | `NONE` | raw samples, not averages over intervals | nothing yet; tsdive reads raw samples |
| [stepped](../../reference/glossary.md#stepped) | `stepped no` | the historian stored a sample every scan | `--stepped`, for a tag the historian stores only when the value changes |

The calculation basis changes what a number means. The time-weighted
mean of the demo flow is 63.95 m3/h. In it, the last sample before the
40-minute outage holds for the whole outage and counts forty times as
much as its neighbours; event-weighted, it would count once. tsdive
refuses to mix reads under two bases with
[`IncomparableSamplingError`](../../reference/errors.md#incomparablesamplingerror).

The retrieval mode and the stepped flag say where the samples came from.
They do not change the arithmetic, but they change what a hole in the
timestamps means, and every report states them.

## The digest

The digest is a hash of the four fields. It identifies the contract,
not the data, so two archives read under one contract print the same
digest. Compare digests to check that two reports mean the same thing
by their numbers.

## When to change the defaults

- A tag stored on change, such as a PI tag with compression: pass
  `--stepped`. A hole of up to 50 times the median interval then reads
  as a value that held, and the gap is classed `compression_steady`
  instead of data loss. See [coverage and gap classes](coverage-and-gaps.md).
- An export of interpolated values, such as sampled data from PI
  DataLink: declare `retrieval_mode: INTERPOLATED` in the metadata, so
  every report says the values are interpolated.
- A counter or an event log: pass `--basis EVENT_WEIGHTED`.

The defaults suit an ordinary measurement exported as recorded values.
