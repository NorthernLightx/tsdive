# Conformal test martingale on the 3W window caches

## Summary

The study runs a conformal test martingale alarm over the detector study's two 3W v2.0.0 window caches (dataset manifest `14bc1397d3ee`): 1113 own-history instances (6992 one-hour windows) and 48 onset-aligned instances (288 windows). Per instance the first two windows fit a median and MAD per variable, the third is the calibration window and the rest are the stream; the martingale fires when it reaches 1/delta. On the 537 normal own-history records the alarm fires on 76.3% at delta 0.05 against a bound of 5.0%, and the worst-baseline rule fires on 67.2%; when the same scores are shuffled across the calibration and stream the alarm fraction is 0.04% (mean over 10 seeds). On the aligned bed the alarm fires in the pre window on 47.9% of 48 instances (floor 25.0%) and first fires in a post window on 37.5%. The bound holds when exchangeability is made true by construction and fails on the records as recorded, so a one-hour calibration window is not exchangeable with the hours that follow it on these wells.

## Data

Own-history bed: `data/3w_windows` (cache manifest sha256 `bfff4f2e897b`), 6992 windows over 1113 instances in `windows.parquet` (the manifest counts 1119 instances; 6 have no window). Onset-aligned bed: `data/3w_windows_aligned` (cache manifest sha256 `6f4f097fcd4a`), 288 windows over 48 instances at offsets -5, -4, -3 (baseline), -2 (pre), 0 (post0) and 1 (post1) from the fault onset. Both caches hold, per window and variable, 60 one-minute sub-group medians of the GOOD rows for the six common variables `P-ANULAR`, `P-JUS-CKGL`, `P-MON-CKP`, `P-PDG`, `P-TPT`, `T-TPT`. How the caches are built is in [the detector report](../3w_detectors/REPORT.md).

Groups on the own-history bed, by the window labels:

- normal (538 instances): every window labelled 0; any alarm is a false alarm. Stream length median 1 window, maximum 49.
- mixed (53): the three baseline windows labelled 0 and at least one later window labelled 1. Stream length median 40 windows, maximum 209.
- positive baseline (54): a baseline window labelled 1, so the fit or the calibration already holds fault rows. The alarm rate is reported and no false-alarm or detection rate.
- short (468): fewer than 4 windows, a design refusal.

The aligned bed is one group of 48 instances; its pre window carries the false-alarm rate and its post windows the detection rate.

## Method

Per instance and bed:

1. Windows are ordered by `window_index` (own history) or `onset_offset` (aligned). Fewer than 4 windows is a design refusal.
2. Fit: `mad_baseline` on the finite sub-group medians of the first 2 windows, per variable (at most 120 values). A variable with under 30 values (`InsufficientQuality`) or a zero MAD is unusable; an instance with no usable variable is a refusal.
3. Nonconformity of a sub-group: the largest |x - centre| / scale over the usable variables with a finite value at that minute. A minute with no finite variable is dropped.
4. Calibration scores: the third window's sub-groups; fewer than 30 finite scores is a refusal. Stream scores: windows four onward, in order.
5. `conformal`: `conformal_p_values(calibration, stream, online=True)`, `mixture_martingale` over epsilons [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], `martingale_alarm` at each delta. The alarm minute maps to its window.
6. `worst_baseline`: the window score is its largest sub-group nonconformity; the threshold is `worst_baseline_threshold` over the three baseline windows (in sample, as in the detector study); the alarm is the first stream window where `fires`. Floor `far_floor(3)` = 0.250.
7. `clock`: the same martingale with nonconformity = position in the record (calibration 0 to 59, stream counting on). Reads no sensor value.
8. `permutation`: calibration and stream scores concatenated, shuffled with `numpy.random.default_rng(seed)` for seeds 0 to 9, split back at the calibration length, then rule 5. Exchangeability then holds by construction.
9. `reversed`: rule 5 with the first stream window as the calibration and the calibration window as the stream (a first stream window with fewer than 30 finite scores is a refusal). The outcome is `alarm` or `none`; no label is read. The scores are the same as in rule 5, so a forward rate far above the reversed one says the score grows with distance from the fit hours.

Outcome per instance and rule: the first alarm only. An alarm in a window labelled 1 is a detection (aligned: post0 or post1), in a window labelled 0 a false alarm (aligned: pre), and no crossing is `none`. Rates are over the scored instances of the group. On the aligned bed an instance whose alarm falls in the pre window is a false alarm and is not also a detection; the column `crossed by end of post1` counts an alarm in any of the three stream windows.

Commands:

```
uv run python examples/studies/3w_conformal/run_conformal.py
uv run python examples/studies/3w_conformal/make_report.py --update-benchmarks
```

The run takes 21.5 s. Two runs write byte-identical CSVs; the study checked this with a second run into a scratch directory and `cmp` on every file.

## Results

### Own-history bed

`alarm rate` is the share of scored instances with any alarm. FAR is the share whose alarm falls in a window labelled 0, detect the share whose alarm falls in a window labelled 1, delay the alarm window minus the first fault window. `median max log10 M` is the median over instances of the largest log10 martingale value on the stream; the alarm lines are 1.30 at delta 0.05 and 2.00 at delta 0.01.

| group | rule | delta | scored | alarm rate | FAR | detect | median delay (windows) | median max log10 M |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| normal | conformal martingale | 0.05 | 537 of 538 | 76.3% | 76.3% |  |  | 13.9 |
| normal | conformal martingale | 0.01 | 537 of 538 | 73.9% | 73.9% |  |  | 13.9 |
| normal | worst-baseline threshold |  | 537 of 538 | 67.2% | 67.2% |  |  |  |
| normal | clock control (position, no sensor read) | 0.05 | 537 of 538 | 100.0% | 100.0% |  |  | 50.7 |
| normal | clock control (position, no sensor read) | 0.01 | 537 of 538 | 100.0% | 100.0% |  |  | 50.7 |
| normal | permutation check (scores shuffled) | 0.05 | 537 of 537 | 0.04% |  |  |  |  |
| normal | permutation check (scores shuffled) | 0.01 | 537 of 537 | 0.00% |  |  |  |  |
| mixed | conformal martingale | 0.05 | 53 of 53 | 100.0% | 71.7% | 28.3% | 0 | 868.7 |
| mixed | conformal martingale | 0.01 | 53 of 53 | 100.0% | 71.7% | 28.3% | 1 | 868.7 |
| mixed | worst-baseline threshold |  | 53 of 53 | 96.2% | 71.7% | 24.5% | 0 |  |
| mixed | clock control (position, no sensor read) | 0.05 | 53 of 53 | 100.0% | 92.5% | 7.5% | 0 | 4076.3 |
| mixed | clock control (position, no sensor read) | 0.01 | 53 of 53 | 100.0% | 92.5% | 7.5% | 0 | 4076.3 |
| mixed | permutation check (scores shuffled) | 0.05 | 53 of 53 | 0.19% |  |  |  |  |
| mixed | permutation check (scores shuffled) | 0.01 | 53 of 53 | 0.00% |  |  |  |  |
| positive baseline | conformal martingale | 0.05 | 54 of 54 | 61.1% |  |  |  | 5.8 |
| positive baseline | conformal martingale | 0.01 | 54 of 54 | 57.4% |  |  |  | 5.8 |
| positive baseline | worst-baseline threshold |  | 54 of 54 | 57.4% |  |  |  |  |
| positive baseline | clock control (position, no sensor read) | 0.05 | 54 of 54 | 100.0% |  |  |  | 50.7 |
| positive baseline | clock control (position, no sensor read) | 0.01 | 54 of 54 | 100.0% |  |  |  | 50.7 |
| positive baseline | permutation check (scores shuffled) | 0.05 | 54 of 54 | 0.18% |  |  |  |  |
| positive baseline | permutation check (scores shuffled) | 0.01 | 54 of 54 | 0.00% |  |  |  |  |

FAR of the normal group by stream length, at delta 0.05:

| stream windows | instances | conformal FAR (delta 0.05) | worst-baseline FAR | clock FAR (delta 0.05) |
|---|---:|---:|---:|---:|
| 1 | 505 | 75.4% | 66.5% | 100.0% |
| 2 to 5 | 13 | 84.6% | 61.5% | 100.0% |
| 6 or more | 19 | 94.7% | 89.5% | 100.0% |

### Onset-aligned bed

The detector-study rows come from its frozen `results/aligned_summary.csv` (variant `all`) and are not recomputed. Their `detect post0` and `detect post1` are the share of post windows above the instance's threshold, whatever the pre window did; this study's rows count the first alarm only, so a pre alarm removes the instance from `detect`. `over floor` is FAR on pre minus 0.250.

| rule | delta | scored | crossed by end of post1 | FAR on pre | over floor | detect post0 | detect post1 | median max log10 M | source |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| conformal martingale | 0.05 | 48 of 48 | 85.4% | 47.9% | +0.229 | 33.3% | 37.5% | 32.8 | this study, tsdive 0.2.0 @ `5d40975e` |
| conformal martingale | 0.01 | 48 of 48 | 81.2% | 45.8% | +0.208 | 33.3% | 35.4% | 32.8 | this study, tsdive 0.2.0 @ `5d40975e` |
| worst-baseline threshold |  | 48 of 48 | 89.6% | 52.1% | +0.271 | 33.3% | 37.5% |  | this study, tsdive 0.2.0 @ `5d40975e` |
| clock control (position, no sensor read) | 0.05 | 48 of 48 | 100.0% | 100.0% | +0.750 | 0.0% | 0.0% | 182.6 | this study, tsdive 0.2.0 @ `5d40975e` |
| clock control (position, no sensor read) | 0.01 | 48 of 48 | 100.0% | 100.0% | +0.750 | 0.0% | 0.0% | 182.6 | this study, tsdive 0.2.0 @ `5d40975e` |
| permutation check (scores shuffled) | 0.05 | 48 of 48 | 0.00% |  |  |  |  |  | this study, tsdive 0.2.0 @ `5d40975e` |
| permutation check (scores shuffled) | 0.01 | 48 of 48 | 0.00% |  |  |  |  |  | this study, tsdive 0.2.0 @ `5d40975e` |
| MAD screen, own history |  | 48 of 48 |  | 62.5% | +0.375 | 81.2% | 91.7% |  | detector study, tagledger 0.1.0 @ `edf2b4ad` |
| SPC individuals rules, own history |  | 48 of 48 |  | 43.8% | +0.188 | 77.1% | 85.4% |  | detector study, tagledger 0.1.0 @ `edf2b4ad` |

### Forward and reversed order

The forward rate is the conformal rule's alarm rate over the whole stream (the `alarm rate` column above); the reversed rate calibrates on the first stream window and streams the calibration window, on the same scores.

| bed | group | delta | scored (reversed) | forward alarm rate | reversed alarm rate | reversed median max log10 M |
|---|---|---:|---:|---:|---:|---:|
| own history | normal | 0.05 | 537 of 538 | 76.3% | 13.2% | -0.3 |
| own history | normal | 0.01 | 537 of 538 | 73.9% | 11.0% | -0.3 |
| own history | mixed | 0.05 | 53 of 53 | 100.0% | 7.5% | -0.3 |
| own history | mixed | 0.01 | 53 of 53 | 100.0% | 5.7% | -0.3 |
| own history | positive baseline | 0.05 | 54 of 54 | 61.1% | 7.4% | -0.3 |
| own history | positive baseline | 0.01 | 54 of 54 | 57.4% | 5.6% | -0.3 |
| aligned | aligned | 0.05 | 48 of 48 | 85.4% | 8.3% | -0.3 |
| aligned | aligned | 0.01 | 48 of 48 | 81.2% | 6.2% | -0.3 |

### Permutation check

Each seed shuffles every scored instance's calibration and stream scores together and reruns the conformal rule; the alarm fraction is over the group's instances.

| bed | group | delta | instances | seeds | mean alarm fraction | min | max |
|---|---|---:|---:|---:|---:|---:|---:|
| aligned | aligned | 0.01 | 48 | 10 | 0.00% | 0.00% | 0.00% |
| aligned | aligned | 0.05 | 48 | 10 | 0.00% | 0.00% | 0.00% |
| own history | mixed | 0.01 | 53 | 10 | 0.00% | 0.00% | 0.00% |
| own history | mixed | 0.05 | 53 | 10 | 0.19% | 0.00% | 1.89% |
| own history | normal | 0.01 | 537 | 10 | 0.00% | 0.00% | 0.00% |
| own history | normal | 0.05 | 537 | 10 | 0.04% | 0.00% | 0.19% |
| own history | positive baseline | 0.01 | 54 | 10 | 0.00% | 0.00% | 0.00% |
| own history | positive baseline | 0.05 | 54 | 10 | 0.18% | 0.00% | 1.85% |

### Refusals

| bed | refusal | instances | instances in cache |
|---|---|---:|---:|
| own history | design | 468 | 1113 |
| own history | no usable variable | 1 | 1113 |
| aligned | none | 0 | 48 |

Usable variables per scored instance (median 3 of 6 on the own-history bed). A variable becomes unusable when its MAD over the two fit windows is zero, which happens when the rounded sub-group medians repeat one value.

| usable variables | own history instances | aligned instances |
|---:|---:|---:|
| 1 | 59 | 0 |
| 2 | 46 | 8 |
| 3 | 235 | 7 |
| 4 | 111 | 8 |
| 5 | 149 | 21 |
| 6 | 44 | 4 |

## Discussion

On the 537 normal records the martingale reaches 1/delta on 76.3% at delta 0.05 and 73.9% at delta 0.01. The bound says at most 5.0% and 1.0%. The permutation check on the same scores gives 0.04% and 0.00%, under the bound, and no (bed, group, delta, seed) cell exceeds 1.89%. The code keeps the bound when the scores are exchangeable; the records are not. The clock control, which reads position alone, fires on 100.0% of the same records with a median peak of 10^50.7: every new minute is the largest position so far, so its p-value is the smallest available and the martingale grows at every step. A record whose level drifts between hours does the same to the sensor scores, at a slower rate (median peak 10^13.9).

The worst-baseline rule fires on 67.2% of the normal records, 76.3% for the martingale at delta 0.05. Most normal records are one stream window long (median 1), so both rates are mostly the chance that the fourth hour exceeds the first three; the length table shows how the rates move with longer streams. The worst-baseline floor 25.0% is a per-window statement and the martingale bound 5.0% a per-record one; neither rule reaches its own reference on this bed.

On the 53 mixed records the martingale fires before the fault on 71.7% and first fires inside a fault window on 28.3% (median delay 0 windows); the worst-baseline rule gives 71.7% and 24.5%, the clock 92.5% and 7.5%. With a median of 40 stream windows and the first fault at a median of window 10 in the record, an alarm that fires eventually fires before the fault more often than in it.

On the aligned bed the martingale fires in the pre window on 47.9% of 48 instances at delta 0.05 (+0.229 over the 25.0% floor, 52.1% for the worst-baseline rule on the same scores) and first fires in post0 on 33.3%, by the end of post1 on 37.5%; by the end of post1 it has crossed at some point on 85.4%. The permutation check gives 0.00%. The detector study's MAD screen fires on 62.5% of the pre windows and the SPC rules on 43.8% under the worst-baseline rule. The clock fires in the pre window on every instance (100.0%).

Reversing the order separates drift from spread. On the 505 normal records with one stream window the forward rule (calibrate on hour 3, stream hour 4) fires on 75.4% at delta 0.05; the reversed rule (calibrate on hour 4, stream hour 3) fires on 13.5% with the same fit and the same scores. Hour-to-hour spread alone would give near-equal rates in both directions. The asymmetry says the nonconformity grows with distance from the fit hours: the level moves away from the centre fitted on hours 1 and 2, so hour 4 outscores hour 3 and not the other way round. The clock row is the extreme case of the same mechanism, a score that grows with position by construction (100.0% forward). On the aligned bed the forward rule fires in the pre window on 47.9% and the reversed rule on 8.3%.

What the numbers support: `conformal_p_values`, `mixture_martingale` and `martingale_alarm` hold their false-alarm bound when the calibration and stream scores are exchangeable, and a 3W own-history record scored this way is not exchangeable at the one-hour scale, so the bound does not transfer to it. What they do not support: a detection claim for the steady fault state (the aligned positives are the first two hours of a transient), a claim about the short instances, or a comparison of the bound with the floor as if they measured the same thing.

## Limits and further work

- Exchangeability is the assumption and the records break it. Detrending the scores (a residual against a slow fit, or a difference between consecutive minutes) before the conformal step, or a longer calibration window, are the untried settings that could restore it. Each is a different study.
- The nonconformity is the maximum robust z over the usable variables; with a median of 3 usable variables the score often reads one or two tags.
- The fit and the calibration are one hour each, and `mad_baseline` on 120 sub-group medians is a small fit. A rounded tag gives a zero MAD and drops out; on this bed that is the only refusal of the sensor path.
- The aligned bed is 48 instances on 21 wells and the positives are transient onsets; no group holdout applies because nothing is fitted across instances.

## Files

| path | what |
|---|---|
| `run_conformal.py` | scores both caches, writes `results/` |
| `make_report.py` | this report and `results/benchmarks_section.md` |
| `results/per_instance.csv` | one row per (bed, instance, rule, delta): alarm window, sub-group, label, outcome, delay, peak log10 martingale, refusal |
| `results/summary.csv` | one row per (bed, group, rule, delta) |
| `results/permutation.csv` | alarm fraction per (bed, group, delta, seed) |
| `results/benchmarks_section.md` | the BENCHMARKS.md REAL section |
| `results/run.json` | code head, tsdive version, cache and dataset manifest hashes, parameters, group and refusal counts, wall seconds |

This study: tsdive 0.2.0 @ `5d40975e`. Detector-study rows: tagledger 0.1.0 @ `edf2b4ad`.
