# Reading the SPC chart

`tsdive spc` runs three SPC rules over a window, with limits from a
baseline. This is the report on the demo flow, then every line.

```tsdive
tsdive spc data/demo/fic101_demo.parquet \
    --baseline 2024-03-30T20:00:00Z/2024-03-31T01:00:00Z \
    --window 2024-03-31T01:00:00Z/2024-03-31T06:00:00Z
```

## Line by line

- `demo:FIC101.PV  37 rule hits in 300 samples`: hits of all three rules
  over the window's GOOD samples. One sample can hit more than one rule.
- `baseline` and `window`: the spans, as in the screen.
- `limits    center 62.32   sigma 0.5488   lcl 60.67   ucl 63.97`: the
  center line, sigma, and the lower and upper control limits at 3 sigma,
  in m3/h.
- `basis     individuals 3-sigma`: an individuals chart, one point per
  sample.
- `BEYOND_3SIGMA  29`: samples outside the limits. A lone hit gives the
  time, the value and the limits it broke, such as `value 100 outside
  [60.67, 63.97]`. Consecutive hits print as one run, here
  `2024-03-31 02:01:00Z -> 02:29:00Z   29 samples`.
- `RUN_9_SAMESIDE  6`: runs of 9 samples on one side of the center, each
  reported at its ninth sample.
- `TREND_6  2`: runs of 6 samples rising or falling, reported from the
  sixth sample on.

Each section lists its first five runs and counts the rest, such as
`(+1 more run)`. A rule that found nothing prints its name and 0.

## What to do next

- `BEYOND_3SIGMA` hits in one long run, here 29 samples from 02:01:
  profile that stretch and read its Range section for clipping; see
  [clipping and censoring](../concepts/clipping-and-censoring.md).
- `RUN_9_SAMESIDE` hits with few `BEYOND_3SIGMA`: a small shift of the
  operating point. `compare` measures its size in sigmas.
- `TREND_6` hits in a row: a ramp or a drift. Check whether an operator
  moved the setpoint.
- The rules, and why sigma comes from the MAD: [SPC rules](../concepts/spc-rules.md).
