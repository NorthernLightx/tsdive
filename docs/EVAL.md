# Evaluation protocol

This page gives the rules a published detector number follows and the
`tsdive.eval` function that carries each rule.

```python
from tsdive.eval import (
    GroupLeakage, clock_control, conformal_p_values, far_floor, fires,
    group_holdout, martingale_alarm, mixture_martingale, over_floor,
    power_martingale, ranking_metrics, worst_baseline_threshold,
)
```

## Rules

1. **Hold out whole groups.** Every window of a group (a well, an
   instance) sits in one fold. `group_holdout(frame, group_col=...,
   stratum_col=..., n_folds=..., seed=...)` builds the folds and
   `GroupSplit.leakage_check()` raises `GroupLeakage` when a group
   reaches two folds. The check runs before the split is returned.

2. **Publish a clock control beside every detector.**
   `clock_control(record_position)` scores a window by how far into its
   own record it sits and reads no sensor value. Its row shows how much
   of a detector's AUC is the detector and how much is time.

3. **Report a one-class fold as a refusal.** `ranking_metrics(y, scores)`
   returns ROC-AUC, PR-AUC and precision at recall 0.5, rounded to four
   decimals by `to_dict()`. When the fold carries one class only, every
   metric is `None` and `refusal` names the cause. The refusal is a row
   in the results, with the same weight as a number.

4. **Set the alarm at the worst baseline score.**
   `worst_baseline_threshold(baseline_scores)` is the largest score the
   tool gave the instance's own baseline windows, and `fires(score,
   threshold)` is true strictly above it.

5. **Read every false-alarm rate against the floor.** A window
   exchangeable with `k` baseline windows exceeds their maximum with
   probability `1 / (k + 1)`, so `far_floor(k)` is the false-alarm rate
   a tool that reads nothing scores, and `over_floor(rate, k)` is the
   gap a measured rate leaves above it. With three baseline windows the
   floor is 25 percent.

6. **Publish an alarm with a stated false-alarm bound.**
   `conformal_p_values(calibration, stream)` gives each stream score a
   p-value against the calibration scores and the stream so far, with
   ties counted against the new score so the value is conservative.
   `mixture_martingale(p_values)` turns them into the log of a test
   martingale (`power_martingale` is one component), and
   `martingale_alarm(log_values, delta)` fires at the first index where
   it reaches `1 / delta`. Ville's inequality bounds the probability of
   ever reaching that line by `delta`, so the bound covers the record
   over its whole run and not each window. It holds when the calibration
   and stream scores are exchangeable, and any fitted centre and scale
   that produce the scores come from a fit set disjoint from both. A
   trending record breaks exchangeability, and the clock control shows
   how far.

## Worked example

The 3W detector study applies rules 1 to 5, holds out by well, and
prints the clock control and the floor in its tables:
[examples/studies/3w_detectors/REPORT.md](../examples/studies/3w_detectors/REPORT.md).
The 3W conformal study applies rule 6 beside rules 2, 4 and 5 and
reports the permutation check that makes exchangeability hold:
[examples/studies/3w_conformal/REPORT.md](../examples/studies/3w_conformal/REPORT.md).
Reproduction steps are in [DATA.md](DATA.md).
