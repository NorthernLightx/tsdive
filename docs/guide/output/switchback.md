# Reading the switchback analysis

`tsdive switchback analyze` measures the difference between settings A
and B on a target tag, under the random schedule a plan fixed before
the trial. This is the analysis of the demo trial, where setting B adds
0.25 degC to the outlet temperature, then every line.

```tsdive lines=1
tsdive switchback plan --window 2024-06-03T00:00:00Z/2024-06-04T00:00:00Z \
    --block PT1H --washout PT15M --seed 7 -o plan.json
```

```tsdive
tsdive switchback analyze data/switchback_demo/*.parquet --plan plan.json \
    --target TI201.PV --covariate FI200.PV --covariate TT001.PV
```

## Headline and plan

- `demo:TI201.PV   B - A +0.6024 degrees Celsius   p 0.154`: the target,
  the direct estimate of B minus A, and its
  [randomization p-value](../../reference/glossary.md#randomization-p-value).
- `plan      digest 66da65ede04f verified   seed 7`: the plan file
  matches its digest, so its schedule is the one drawn before the trial.
- `schedule` and `blocks`: 24 blocks of 1 h, 12 of each setting, the
  first 15 min of each left out as [washout](../../reference/glossary.md#washout).
- `design ... smallest p 0.000999`: the p-value is read over the
  observed schedule plus 1000 drawn ones, so it cannot go below 1/1001.
- `units` and `quality`: the target's unit, its GOOD share, and its
  censoring verdict.

## Difference in means

- `estimate  +0.6024 degrees Celsius   p 0.154`: the mean of the kept B
  samples minus the mean of the kept A samples. About 154 in 1000
  reshuffled schedules give a difference at least this large, so this
  estimate alone does not show an effect.
- `95%       [-0.2223, +1.500]`: the shifts the test does not reject.
  Read it as a confidence range. It includes 0.
- `kept      A 540   B 540   per block 45 to 45`: samples kept after the
  washout, 45 of every block's 60.

## Adjusted

- `(OLS on 2 covariates)` and the covariate names: the feed flow and the
  ambient temperature, disturbances the setting cannot move.
- `estimate  +0.2847 degrees Celsius   p 0.000999`: B minus A after the
  fit removes what the covariates explain. p is at its floor: no drawn
  schedule gave a difference this large.
- `95%       [+0.2214, +0.3444]`: a range of 0.12 degC around the
  estimate, and it holds the true 0.25.

## Assumptions

The three lines at the end state what the numbers rest on: the schedule
was followed, and the process settled within the washout. "By intended
assignment" means blocks count by the setting the plan gave them, even
if an operator switched late.

## Direct or adjusted

The direct estimate needs only the randomization. The adjusted one also
needs every covariate to be a disturbance the setting cannot move. Here
the feed flow and the ambient temperature moved the outlet temperature
during the trial, and by chance more of that movement fell in B blocks.
The direct estimate carries it; the adjusted one takes it out and lands
near the true 0.25 degC. Report the adjusted estimate when the
covariates pass that test, and say which covariates you used. `analyze` flags a covariate whose
own B - A difference is itself clear, because the setting moved it.

## For the plant manager

Two sentences carry the result: "Over one day we alternated the old (A)
and new (B) setting every hour, in a random order fixed in advance. With
the new setting the outlet ran 0.28 degC hotter, likely between 0.22
and 0.34 degC, after removing feed and ambient swings; a difference this
large turned up in none of 1000 reshuffled schedules."

The [controller trial how-to](../howto/switchback-trial.md) walks
through planning and running a trial like this one.
