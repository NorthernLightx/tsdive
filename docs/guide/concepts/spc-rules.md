# SPC rules

`tsdive spc` draws an
[individuals chart](../../reference/glossary.md#individuals-chart): one
point per sample, a center line, and limits 3 sigma either side. It
takes the center and the sigma from a baseline, the same way the
[MAD screen](baselines.md) does, and runs three
[SPC rules](../../reference/glossary.md#spc-rule) over the window.

## The three rules

| rule | fires when | points at |
|---|---|---|
| `BEYOND_3SIGMA` | a sample lies outside the 3-sigma limits | a spike, a step, a saturated transmitter |
| `RUN_9_SAMESIDE` | 9 samples in a row sit on one side of the center | a shift too small to cross the limits |
| `TREND_6` | 6 samples in a row each rise, or each fall | a drift, a ramp, a fouling trend |

These are rules 1, 2 and 3 of the Nelson set. Every rule reports its
own hits, and a rule with none reports 0, so a 0 means the rule ran and
found nothing.

```tsdive
tsdive spc data/demo/fic101_demo.parquet \
    --baseline 2024-03-30T20:00:00Z/2024-03-31T01:00:00Z \
    --window 2024-03-31T01:00:00Z/2024-03-31T06:00:00Z
```

The limits are the screen's: center 62.32 m3/h, sigma 0.5488, limits
60.67 and 63.97. `BEYOND_3SIGMA 29` is the half hour at 100 m3/h.
`RUN_9_SAMESIDE 6` marks runs where the flow sat above or below 62.32
for 9 minutes or more; the demo flow swings slowly around its median,
so some runs are expected. `TREND_6 2` marks two stretches of six
minutes of steady rise or fall.

A run of 20 samples above the center fires `RUN_9_SAMESIDE` once, at its
ninth sample. A long rise fires `TREND_6` at every sample from the sixth
on.

## Sigma comes from the MAD

The chart's sigma is 1.4826 times the MAD of the baseline, not the
average moving range of the textbook individuals chart. For normal noise
the two agree. When the baseline holds a few spikes, the MAD ignores
them and the limits stay tight.

## Reading the hits

- Hits of `BEYOND_3SIGMA` in a censored window are the clipped stretch;
  read the Range section of the profile first.
- Many `RUN_9_SAMESIDE` hits and no `BEYOND_3SIGMA` point at a small
  shift of the operating point. Check whether the regime changed.
- `TREND_6` on a slow loop is common, because consecutive samples of a
  60 s tag are not independent. Read it together with the other two.

## What to do

- Use `spc` to see which kind of departure a window holds. Use `screen`
  to count samples outside the limits, and `compare` to measure how far
  the level and the spread moved.
- For a JSON list of every hit, pass `--json`; `SpcAnalysis.frame` holds
  one row per hit in Python.
