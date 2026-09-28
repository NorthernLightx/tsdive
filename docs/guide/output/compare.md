# Reading the compare report

`tsdive compare` measures what changed between two periods over every
tag of a unit. This is the report on the demo flow and temperature,
before 20:00 to 23:00, after 04:00 to 06:00, then every line.

```tsdive
tsdive compare data/demo/fic101_demo.parquet data/demo/tic101_demo.parquet \
    --before 2024-03-30T20:00:00Z/2024-03-30T23:00:00Z \
    --after 2024-03-31T04:00:00Z/2024-03-31T06:00:00Z
```

## Headline and periods

- `demo  2 tags   refused 0   pairs 1 of 1 clearing`: the source, the
  tags compared, the tags whose read was refused, and the pairs whose
  interval excludes 0.
- `before` and `after`: the two periods and their lengths.
- `grid      rate 60 s (declared)   coverage 1.000 -> 1.000`: the rate
  the pair and joint tables align on, and the share of grid cells filled
  in each period.

A refused table prints its reason under the headline, such as a missing
`sample_rate_s`.

## Tags that changed

Every tag is listed, ranked by the size of its level shift.

- `TIC101.PV  ok  +0.8  x6.0  62%  2024-03-31 04:18:00Z`: after-period
  quality is `ok`; the median rose 0.8 before-period sigmas; the MAD is
  six times the before MAD; a screen built on the before period flags 62%
  of the after samples; and the first change point after 04:00 is at
  04:18.
- `FIC101.PV  ok  +0.4  x0.5  0%  2024-03-31 04:00:00Z`: the flow moved
  0.4 sigma, got quieter, and no sample leaves the before limits. It
  changed little.

## Pairs that decoupled

- `FIC101.PV ~ TIC101.PV    0.69    0.17   -0.53  [-0.76, -0.26]`: the
  minute-to-minute correlation of flow and temperature was 0.69 before
  and 0.17 after. The drop of 0.53 has a 95% interval from -0.76 to
  -0.26, which excludes 0, so the pair clears.

## Joint structure

- `components 1 of 2   explained 0.9839 -> 0.2643 on after`: the PCA
  fitted on the before period keeps one component, which carries 98% of
  the before variance and 26% of the after variance.
- `rows 121   T2 breaches 22   SPE breaches 108`: after rows, and those
  above the before period's T2 and SPE limits.
- `SPE contributors   TIC101.PV 79%   FIC101.PV 21%`: the temperature
  carries four fifths of the broken structure.

## What the demo says

The temperature got noisier and stopped following the flow from about
04:00; the flow barely changed. The report does not say why. A fouled
thermowell, a loose sensor and a new heat source give the same tables.
[Compare's three tables](../concepts/compare.md) explains each column.

## What to do next

- Start with the top row of the tag table and its `changed at` time.
- A clearing pair: check the process link between the two tags at that
  time.
- The tables as data: `--json`, or `CompareAnalysis.to_dict()` in
  Python; `CompareAnalysis.frame` holds the tag table.
