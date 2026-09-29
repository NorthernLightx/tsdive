# Baselines and the MAD screen

`screen`, `spc`, `mspc` and `compare` judge a window against a
[baseline](../../reference/glossary.md#baseline): a stretch of history
you trust. This page covers how the MAD screen turns a baseline into
limits, what its caveat means, and how to screen per operating regime.

## Center and sigma from the baseline

The screen reads the GOOD samples of the baseline and takes two numbers
from them:

- the **center**, their median: the middle value, which a few wild
  samples cannot pull;
- the **scale**, 1.4826 times their
  [MAD](../../reference/glossary.md#mad), the median distance of the
  samples from the median. For normally distributed noise it equals the
  standard deviation, and it is the
  [sigma](../../reference/glossary.md#sigma) every limit is counted in.

The limits sit `k` sigmas either side of the center, 3 by default. A
window sample outside them is
[flagged](../../reference/glossary.md#flagged):

```tsdive
tsdive screen data/demo/fic101_demo.parquet \
    --baseline 2024-03-30T20:00:00Z/2024-03-31T01:00:00Z \
    --window 2024-03-31T01:00:00Z/2024-03-31T06:00:00Z
```

The baseline's median is 62.32 m3/h and its sigma 0.5488 m3/h, so the
limits are 62.32 - 3 x 0.5488 = 60.67 and 62.32 + 3 x 0.5488 = 63.97
m3/h. 29 of the 300 GOOD window samples fall outside, all of them in the
half hour at 100 m3/h.

A standard deviation over the same baseline would count every spike in
its own spread. The MAD does not, which keeps the limits tight when the
history holds a few bad samples.

## Provisional: one baseline for every regime

The `caveat provisional` line says the screen used one baseline for the
whole window. That is right only if the process runs at one operating
point. A throughput step, a recipe change or a grade change moves the
median, and every sample after it reads as flagged. The flags then say
"the operating point moved", not "the tag misbehaved".

`--method moving-range` estimates the scale from the mean jump between
successive samples instead. It carries its own caveat: on a stepping
process it measures the steps as well as the noise.

## Screening per regime

A [regime](../../reference/glossary.md#regime) is a stretch at one
operating point. When a [MODE tag](../../reference/glossary.md#mode-tag)
records it, such as a recipe step, `screen --mode` builds one baseline
per regime and screens each window sample against its own regime's
baseline. When no MODE tag exists, `segment` finds the regimes in the
samples and writes them as one:

```tsdive
tsdive segment data/demo/fic101_demo.parquet \
    --window 2024-03-30T20:00:00Z/2024-03-31T02:00:00Z --mode-out modes.parquet
tsdive screen data/demo/fic101_demo.parquet --mode modes.parquet \
    --baseline 2024-03-30T20:00:00Z/2024-03-31T01:00:00Z \
    --window 2024-03-31T01:00:00Z/2024-03-31T02:00:00Z
```

The segment found three regimes before 02:00, with medians of 62.54,
61.66 and 62.56 m3/h. Each has its own center and scale, so the window
hour from 01:00, all of it in regime S3, is judged against S3's scale of
0.2301 m3/h rather than the whole baseline's 0.5488.

Every regime in the window must appear in the baseline with enough GOOD
samples, else the screen raises
[`RegimeTooSparse`](../../reference/errors.md#regimetoosparse).

## Rules for a baseline

- It holds at least 30 GOOD samples.
- It holds no clipped sample. See
  [clipping and censoring](clipping-and-censoring.md).
- Its GOOD values spread. A baseline whose MAD or moving-range scale is
  0 raises
  [`ZeroSpreadBaseline`](../../reference/errors.md#zerospreadbaseline).
- It does not overlap the window.

A baseline that breaks the first two raises
[`InsufficientQuality`](../../reference/errors.md#insufficientquality);
an overlap is a usage error.

## What to do

- Pick a baseline at the same operating point as the window, from a
  period the operators call normal.
  [Choose a clean baseline window](../howto/baseline-window.md) walks
  through it.
- Read a high flagged share together with the caveat: if the regime
  changed, screen per regime before you call it a fault.
