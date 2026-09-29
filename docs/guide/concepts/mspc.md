# MSPC: T2, SPE and contributors

A single-tag chart misses a fault that shows only in how tags move
together: a temperature that stops following its flow while both stay
inside their own limits. `tsdive mspc` watches two or more tags at once
with a [PCA](../../reference/glossary.md#pca) model of their joint
behaviour, and reports two statistics per timestamp.

## The model

`mspc` lines the archives up on one UTC grid at their declared
`sample_rate_s`, or at `--rate-s`. It refuses to interpolate: if less
than 0.95 of the grid cells are filled, it raises
[`MspcAlignmentError`](../../reference/errors.md#mspcalignmenterror).
On the baseline rows it fits a PCA: it finds the directions in which
the tags vary together and keeps the fewest components that carry 0.95
of the variance (`--variance`). The model works in the tags' own units,
centred on the baseline mean, so a tag with a larger spread weighs more.

## Two statistics per row

- [T2](../../reference/glossary.md#t2) measures how far a row sits from
  the baseline's center along the kept components, in units of the
  baseline's own variation. A big T2 is a move the correlations allow,
  only bigger: both tags up together, further than before.
- [SPE](../../reference/glossary.md#spe) measures how far a row sits off
  the kept components, the part the model cannot explain. A big SPE is a
  broken correlation: one tag moves and its partner does not.

Each has a limit, the 0.99
[empirical quantile](../../reference/glossary.md#empirical-quantile) of
its baseline values (`--quantile`). A row above the limit is a breach.
The limits make no distributional assumption, so about 1 in 100 baseline
rows breach by construction.

```tsdive
tsdive mspc data/demo/fic101_demo.parquet data/demo/tic101_demo.parquet \
    --baseline 2024-03-30T20:00:00Z/2024-03-30T23:00:00Z \
    --window 2024-03-31T04:00:00Z/2024-03-31T06:00:00Z
```

- `components 1 of 2   explained 0.9839`: one component carries 98% of
  the baseline variance, because the temperature follows the flow.
- `limits    T2 4.052   SPE 0.03662`: the breach thresholds.
- `T2 breaches 22   SPE breaches 108   of 121 rows`: after 04:00 the
  temperature stops tracking the flow. 108 of 121 rows break the
  correlation, and 22 sit far along it.

## Contributors

[Contributors](../../reference/glossary.md#contributors) name the tags
that carry the breaches, each with its share. With three tags or fewer
every tag would be named, so `mspc` prints `not ranked`. `compare`'s
joint table prints the shares for any number of tags: on the demo,
TIC-101 carries 79% of the SPE.

## What to do

- Declare `sample_rate_s` on every archive, or pass `--rate-s`; without
  a grid there is no model.
- Fit the baseline away from outages. A baseline from 20:00 to 01:00
  holds the demo flow's 40-minute hole and leaves 0.867 of the grid
  filled.
- Read SPE breaches as "a relationship broke" and check the tag the
  contributors name: a sensor fault, a fouled exchanger, a valve that
  stopped responding.
