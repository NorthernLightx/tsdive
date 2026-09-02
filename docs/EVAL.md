# Evaluation protocol

This page gives the rules a published detector number follows and the
`tsdive.eval` function that carries each rule.

```python
from tsdive.eval import (
    GroupLeakage, clock_control, far_floor, fires, group_holdout,
    over_floor, ranking_metrics, worst_baseline_threshold,
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

## Worked example

The 3W detector study applies all five rules, holds out by well, and
prints the clock control and the floor in its tables:
[examples/studies/3w_detectors/REPORT.md](../examples/studies/3w_detectors/REPORT.md).
Reproduction steps are in [DATA.md](DATA.md).
