# Reading the MSPC report

`tsdive mspc` watches two or more tags at once with a PCA model fitted
on a baseline. This is the report on the demo flow and temperature,
then every line.

```tsdive
tsdive mspc data/demo/fic101_demo.parquet data/demo/tic101_demo.parquet \
    --baseline 2024-03-30T20:00:00Z/2024-03-30T23:00:00Z \
    --window 2024-03-31T04:00:00Z/2024-03-31T06:00:00Z
```

## Line by line

- `demo:FIC101.PV, demo:TIC101.PV  T2 breaches 70   SPE breaches 108   of 121 rows`:
  the tags in the model, then the window rows whose
  [T2](../../reference/glossary.md#t2) or
  [SPE](../../reference/glossary.md#spe) exceed their limits, out of the
  aligned rows.
- `baseline ... rows 181   coverage 1.000`: grid rows in the baseline
  and the share of grid cells filled. Below 0.95 the model is refused.
- `window ... rows 121   coverage 1.000`: the same for the window.
- `model     rate 60 s (declared)   components 1 of 2   explained 0.9767`:
  the grid rate and where it came from, `declared` from `sample_rate_s`
  or `--rate-s`; the components kept out of the number of tags; and the
  share of baseline variance they carry.
- `limits    T2 3.759   SPE 0.2859   (empirical q0.99)`: each limit is
  the 0.99 quantile of the baseline's own values. Both are in units of
  each tag's baseline standard deviation.
- `contributors  not ranked (2 tags; top-3 would list every one)`: with
  three tags or fewer every tag would be named, so none is ranked.
- `T2 breaches  70` and `SPE breaches  108`: the first breach
  timestamps of each statistic.

A model that keeps as many components as there are tags leaves no
residual. `mspc` then prints `SPE  NOT ASSESSED` with the `--variance`
value that keeps one component fewer, `SPE n/a` as the limit, and no
contributors. The JSON carries `spe_breaches: null` and the reason in
`spe_not_assessed`.

## What the demo says

Before 04:00 one component carries 98% of the variance: the temperature
moves with the flow. After 04:00 the temperature carries a burst of its
own. 108 of 121 rows breach SPE, the correlation broke; 70 breach T2.
The SPE breaches start at 04:00, when the burst starts.

## What to do next

- SPE breaches: a relationship between tags broke. Look at the tag the
  contributors name, or run `compare` for per-tag shares.
- T2 breaches without SPE breaches: the tags moved together further
  than the baseline did, such as a throughput change. Check whether it
  was planned.
- A refusal about coverage: move the baseline off the outage. See
  [MSPC](../concepts/mspc.md).
