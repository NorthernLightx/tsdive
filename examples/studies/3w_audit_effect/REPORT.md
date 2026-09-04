# What the zero-scale and quality checks change on 3W

## Summary

`own_history_baselines` drops an (instance, variable) pair when `mad_baseline` raises `InsufficientQuality` or when the scale it returns is zero. On the 3W own-history cache the zero-scale check drops 690 of the 3870 pairs and the quality check drops 0. Keeping those pairs and scoring a zero scale the way naive code does moves the pooled MAD screen AUC from 0.8665 to 0.8183 and the SPC AUC from 0.7540 to 0.7474, against a clock control at 0.8997. It also gives 68 of the 645 instances an infinite screen threshold, which is why the false-alarm rate on instances with no fault window falls from 73.5% to 69.0% while detection on the instances whose fault arrives later falls from 100.0% to 90.6%. 748 of the 1643 tags that hold one value across their whole archive sit on the six variables this cache carries.

## Data

- 3W v2.0.0, dataset manifest sha256 `14bc1397d3ee`.
- Window cache `data/3w_windows`, manifest `bfff4f2e897b`, 1113 instances and 6992 one-hour windows, each holding 60 one-minute sub-group medians for the six variables every instance shares.
- Groups by label, the same split the conformal study uses: 538 instances with every window labelled 0, 53 with a label-0 prefix and a fault later, 54 with a fault row inside the baseline windows and 468 with 3 windows or fewer, which the design refuses.
- The frozen-tag population is read from `examples/studies/3w_profile/results/per_tag.csv`, the profile study's per-tag table.

Nothing is fitted across instances, so no group holdout applies. The well is carried in every result row, so a later reader can group by it.

## Method

Run the study and render this report:

```
uv run python examples/studies/3w_audit_effect/run_audit_effect.py
uv run python examples/studies/3w_audit_effect/make_report.py --update-benchmarks
```

Each instance's first 3 windows are its baseline, chosen by position and never by label, and the rest are its stream. A pair's centre and scale come from `mad_baseline` over that instance's baseline sub-group medians, the history `own_history_baselines` builds. The MAD screen scores a window by the largest robust z of the window medians over the usable variables; the SPC chart scores it by the rule hits `apply_rules` returns over the sub-group medians, summed over the usable variables. The clock control scores the window's position in its own instance.

Two variants share that code path.

- `checked`: the shipped behaviour. A pair whose baseline raises `InsufficientQuality`, or whose scale is zero, is dropped, and a window with no usable variable is refused.
- `unchecked`: every pair with at least one finite baseline value is kept. Where the scale is zero the MAD screen scores 0 for a window median equal to the centre and `+inf` for any other, and the SPC limits collapse onto the centre, so every sub-group median that differs from it counts as a rule hit. A pair that raises `InsufficientQuality` keeps the median and MAD its finite values give.

The alarm threshold is the largest score the detector gives that instance's own baseline windows, which is in sample by construction, and `fires` is strict, so `far_floor(3)` = 25.0% is the rate a score that reads nothing already reaches. For the ranking metrics a `+inf` score is replaced by one step above the largest finite score, so every infinite window shares the top rank.

The cache holds a historian null sentinel that no float32 can carry, and the study blanks it before either variant runs: 4795 sub-group cells over 80 rows, and 80 window medians. Both variants get the same mask, so it is outside the difference the study measures.

## Results

### What the checks drop

| variant | pairs | scored | dropped, zero scale | dropped, InsufficientQuality | no finite baseline value |
|---|---|---|---|---|---|
| checked | 3870 | 2342 | 690 | 0 | 838 |
| unchecked | 3870 | 3032 | 0 | 0 | 838 |

The quality check drops 0 pairs. A pair on this cache either holds all 180 baseline minutes, or none at all, or one of the 44 partial baselines, and the smallest of those holds 50 finite minutes against the 30 `mad_baseline` requires.

| variable | pairs with a zero scale | one distinct baseline value | more than one |
|---|---|---|---|
| `P-ANULAR` | 55 | 12 | 43 |
| `P-JUS-CKGL` | 43 | 37 | 6 |
| `P-MON-CKP` | 2 | 1 | 1 |
| `P-PDG` | 448 | 436 | 12 |
| `P-TPT` | 88 | 36 | 52 |
| `T-TPT` | 54 | 19 | 35 |
| all | 690 | 541 | 149 |

541 of the 690 zero-scale pairs hold one distinct value across their baseline windows and 149 hold more than one, which is a tag rounded to a lattice coarse enough that half its baseline minutes land on the median.

### Ranking, pooled over every scored stream window

| detector | variant | AUC | stream windows scored | windows scoring +inf |
|---|---|---|---|---|
| MAD screen, max robust z | checked | 0.8665 | 4495 | 0 |
| MAD screen, max robust z | unchecked | 0.8183 | 4495 | 364 |
| SPC individuals rule hits | checked | 0.7540 | 4495 | 0 |
| SPC individuals rule hits | unchecked | 0.7474 | 4495 | 0 |
| clock control (window position in its own instance, no sensor read) | checked | 0.8997 | 4495 | 0 |
| clock control (window position in its own instance, no sensor read) | unchecked | 0.8997 | 4495 | 0 |

### Instances with no fault window

| detector | variant | false-alarm rate per window | over the 25.0% floor | instances raising an alarm | instances with an infinite threshold |
|---|---|---|---|---|---|
| MAD screen, max robust z | checked | 73.5% | +0.484 | 71.2% | 0 |
| MAD screen, max robust z | unchecked | 69.0% | +0.440 | 63.4% | 58 |
| SPC individuals rule hits | checked | 53.4% | +0.284 | 45.4% | 0 |
| SPC individuals rule hits | unchecked | 55.2% | +0.302 | 48.5% | 0 |
| clock control (window position in its own instance, no sensor read) | checked | 100.0% | +0.750 | 100.0% | 0 |
| clock control (window position in its own instance, no sensor read) | unchecked | 100.0% | +0.750 | 100.0% | 0 |

### Instances whose fault arrives later

| detector | variant | detect | median delay (windows) | false-alarm rate before the first fault window | over the 25.0% floor |
|---|---|---|---|---|---|
| MAD screen, max robust z | checked | 100.0% | 0.0 | 87.2% | +0.622 |
| MAD screen, max robust z | unchecked | 90.6% | 0.0 | 83.3% | +0.583 |
| SPC individuals rule hits | checked | 100.0% | 0.0 | 70.5% | +0.455 |
| SPC individuals rule hits | unchecked | 100.0% | 0.0 | 71.4% | +0.464 |
| clock control (window position in its own instance, no sensor read) | checked | 100.0% | 0.0 | 100.0% | +0.750 |
| clock control (window position in its own instance, no sensor read) | unchecked | 100.0% | 0.0 | 100.0% | +0.750 |

### The frozen-tag population

| population | tags |
|---|---|
| measurement tags in the profile | 7337 |
| tags holding one distinct value over the whole archive | 1643 |
| of those, on the six common variables | 748 |
| of those, in an instance the window cache holds | 744 |
| of those, in an instance this design scores | 504 |

## Discussion

The MAD screen loses 0.0482 of AUC when the checks come off, 0.8665 to 0.8183, and its alarm rule changes shape. 68 of the 645 scored instances take an infinite threshold from their own baseline, so nothing they score afterwards can exceed it. Detection on the instances whose fault arrives later drops from 100.0% to 90.6%, and the false-alarm rate on instances with no fault window drops from 73.5% to 69.0% because 58 of those 538 instances can no longer raise an alarm at all.

The SPC chart moves less: AUC 0.7540 to 0.7474, detection 100.0% to 100.0%, and the false-alarm rate on instances with no fault window rises from 53.4% to 55.2%. Collapsed limits produce a finite hit count rather than an infinity, so a zero-scale pair adds hits to the baseline windows and the stream windows alike and the threshold moves with them.

The clock control does not move at all: 0.8997 under both variants, above the MAD screen's 0.8665 either way. The checks change what the detectors read. They leave the position signal this design carries where it was.

The population row bounds how far this reaches. Of the 1643 measurement tags that hold one value across their whole archive, 748 sit on the six common variables, 744 in an instance the cache holds and 504 in an instance this design scores. `P-PDG` carries 652 of them.

## Limits and further work

- The study measures two checks. It says nothing about the float32 mask, the window builder's own refusals, or the profile's flatline verdict, which the detector study does not consult.
- The `unchecked` SPC rule count is an equivalent, not a call into `apply_rules` with zero limits: a zero sigma makes the three-sigma band a point, and the study counts the sub-group medians that differ from the centre.
- The alarm rule reads each instance's own baseline scores as its threshold, which is in sample. That is what an alarm limit set during a learning period is, and it is what makes the false-alarm rate readable, but it is not a held-out threshold.
- The frozen-tag population comes from the profile study's table, which covers 7337 measurement tags across all real instances. The overlap counted here is by instance and variable, not by the window range this cache cut.

## Files

| file | holds |
|---|---|
| `run_audit_effect.py` | the study runner |
| `make_report.py` | this report and the BENCHMARKS section |
| `results/pairs.csv` | one row per (instance, variable, variant) |
| `results/per_record.csv` | the alarm rule and its metrics per instance |
| `results/summary.csv` | one row per (detector, variant, group) |
| `results/run.json` | manifests, code version, parameters, counts, the frozen-tag population |
| `results/benchmarks_section.md` | the BENCHMARKS.md REAL section |
