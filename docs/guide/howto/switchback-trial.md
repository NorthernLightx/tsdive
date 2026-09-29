# Test a controller tuning with a switchback trial

You retuned a temperature loop and want to know whether the outlet runs
differently with the new tuning. Comparing last week with this week
mixes the tuning with everything else that changed: feed, weather,
catalyst age. A [switchback](../../reference/glossary.md#switchback)
trial alternates the old and the new tuning on the same unit in an order
drawn at random before the trial, so those other changes fall on both
settings alike. This guide plans, sizes, runs and reads one on the demo
trial data, where the new setting raises the outlet temperature
TI-201 by 0.25 degC through a lag with a 5-minute time constant.

## 1. Name the settings

Setting A is the current tuning, setting B the new one. The result is
always B minus A, so a positive number means the new tuning runs the
target higher. Name one target tag, here `TI201.PV`, and decide what
"better" means for it before the trial. tsdive tests a difference in
means; to test a difference in variability, see
[the last section](#variability-a-workaround).

## 2. Choose the washout

After each switch the loop needs time to settle. The analysis leaves the
start of every block out; that is the
[washout](../../reference/glossary.md#washout). Use three
[time constants](../../reference/glossary.md#time-constant): a
first-order loop has covered 95% of a step after three. The demo loop
has a 5-minute time constant, so the washout is 15 minutes. If you think
in settling time, the 95% settling time is the washout. Read the time
constant off a setpoint step on the trend: the time to cover 63% of the
step.

## 3. Choose block length and count

The window is cut into [blocks](../../reference/glossary.md#block) of
equal length, and each block runs one setting. The count is the window
divided by the block length, rounded down to an even number, so an odd
count drops the last block. Half the blocks run B.

Two things pull against each other:

- More blocks give more possible schedules and more power, and every
  block is one more A against B comparison.
- Every block loses its washout. In one day with a 15-minute washout,
  2 h blocks keep 105 of 120 minutes, 1 h blocks keep 45 of 60, and
  30-minute blocks keep 15 of 30.

A trial needs at least eight blocks: fewer cannot reach p 0.05 however
large the effect, and the plan is refused with
[`DesignTooSmall`](../../reference/errors.md#designtoosmall).

## 4. Size the trial with the power readout

Give `plan` a history window of the target where nothing was switched,
at least as long as the schedule. It lays 200 random schedules over the
history, adds a shift to the B blocks, and counts how often the analysis
claims a difference. Three block lengths for a one-day trial, with the
day before as history:

```tsdive tail=4
tsdive switchback plan --window 2024-06-03T00:00:00Z/2024-06-04T00:00:00Z \
    --block PT2H --washout PT15M --seed 7 -o plan-2h.json \
    --history data/switchback_demo/ti201.parquet \
    --history-window 2024-06-02T00:00:00Z/2024-06-03T00:00:00Z
tsdive switchback plan --window 2024-06-03T00:00:00Z/2024-06-04T00:00:00Z \
    --block PT1H --washout PT15M --seed 7 -o plan-1h.json \
    --history data/switchback_demo/ti201.parquet \
    --history-window 2024-06-02T00:00:00Z/2024-06-03T00:00:00Z
tsdive switchback plan --window 2024-06-03T00:00:00Z/2024-06-04T00:00:00Z \
    --block PT30M --washout PT15M --seed 7 -o plan-30m.json \
    --history data/switchback_demo/ti201.parquet \
    --history-window 2024-06-02T00:00:00Z/2024-06-03T00:00:00Z
```

Read the table this way:

- `sigma` is the history's spread, 1.4826 times its MAD: 0.4501 degC.
- `shift` is the effect added to the B blocks, in sigmas. Multiply by
  sigma for degC: 0.5 sigma is 0.23 degC.
- `claimed` is the share of the 200 schedules whose 95% interval
  excludes 0. At a shift of 0 it is the false-claim rate, which should
  sit near 0.05; 200 schedules put about 3 points of noise on it. Above
  0 it is the chance of detecting a shift that size.
- `smallest` is the smallest shift detected in at least 80% of the
  schedules.

On this day, 2 h blocks detect a 1-sigma shift (0.45 degC) in 28.5% of
schedules, 1 h blocks in 62%, and 30-minute blocks in 90%. None detects
the 0.25 degC of the demo reliably from the temperature alone. Shorter
blocks help here because the loop settles fast; for a slow loop the
washout eats a short block, and a longer trial is the way to more
blocks.

The readout judges the estimate without covariates. Covariates that
explain part of the noise make the adjusted estimate sharper than the
readout says, as step 7 shows. The readout uses only the first
schedule-length of a longer history.

## 5. Plan the trial and hand it over

Plan the trial you chose, and keep the file: it records the one random
draw the trial must follow.

```tsdive lines=16
tsdive switchback plan --window 2024-06-03T00:00:00Z/2024-06-04T00:00:00Z \
    --block PT1H --washout PT15M --seed 7 -o plan.json
```

The schedule prints in UTC. Convert it to the local time the shift
works in before you hand it over, and give the operators the block
times and settings, not the seed. An existing plan file is never
overwritten, and an edited one no longer matches its digest.

## 6. Choose covariates

A [covariate](../../reference/glossary.md#covariate) explains part of
the target's noise: the outlet temperature follows the feed flow and the
weather. The adjusted estimate removes that part. A covariate is safe
only when the setting cannot move it:

- Safe: feed flow and composition set upstream, ambient temperature,
  cooling water supply temperature.
- Never: the loop's OP, its SP, the manipulated flow, or anything the
  target itself drives. The new tuning moves them, and the fit takes the
  effect away with them.

Declare the covariates before you look at any result, and report them.
Here is what an unsafe one does. A controller output that answers the
temperature, built from the trial data:

```pycon
>>> import numpy as np
>>> import pandas as pd
>>> import tsdive
>>> ti = pd.read_parquet("data/switchback_demo/ti201.parquet")
>>> op = ti.assign(value=(50.0 - 8.0 * (ti["value"] - 180.0)).round(3))
>>> meta = tsdive.TagMeta(identity=tsdive.TagIdentity("demo", "TIC201.OP"),
...                       name="TIC-201 output", unit_raw="%", sample_rate_s=60.0)
>>> _ = tsdive.write_tag("tic201_op.parquet", op, meta)
>>> plan = tsdive.switchback_plan("2024-06-03T00:00:00Z", "2024-06-04T00:00:00Z",
...                               block="PT1H", washout="PT15M", seed=7)
>>> r = tsdive.switchback_analyze(
...     ["data/switchback_demo/ti201.parquet", "tic201_op.parquet"], plan,
...     target="TI201.PV", covariates=["TIC201.OP"])
>>> print(f"direct {r.direct.estimate:+.2f} degC, adjusted {abs(r.adjusted.estimate):.2f} degC")
direct +0.60 degC, adjusted 0.00 degC
>>> r.to_dict()["covariate_checks"][0]["moves_with_setting"]
False

```

The adjusted estimate drops to 0, because the output carries the whole
effect. `analyze` tests each covariate's own B minus A difference and
flags one that moves with the setting, but on one day the output's own
difference is not clear enough to flag. The check guards against the
obvious case; the rule above guards against the rest.

## 7. Analyze and read the result

After the trial, analyze the archives under the plan, with the safe
covariates:

```tsdive
tsdive switchback analyze data/switchback_demo/*.parquet --plan plan.json \
    --target TI201.PV --covariate FI200.PV --covariate TT001.PV
```

The direct estimate, +0.6024 degC with p 0.154, does not show an effect
on its own: the feed flow and the weather moved the outlet too. The
adjusted estimate, +0.2847 degC with a 95% range from +0.2214 to
+0.3444, removes them and holds the true 0.25. [Reading the switchback
analysis](../output/switchback.md) explains every line.

For a plant manager, two sentences carry it: "Over one day we
alternated the old and new tuning every hour, in a random order fixed in
advance. With the new tuning the outlet ran 0.28 degC hotter, likely
between 0.22 and 0.34 degC, after removing feed and weather swings, and
none of 1000 reshuffled schedules gave a difference that large."

## 8. What breaks a trial

- Late switching. Blocks count by the setting the plan gave them. A
  switch 10 minutes late puts 10 minutes of the old tuning into a B block;
  a washout at least as long as the lateness absorbs it. Check the SP,
  OP or mode tag against the schedule after the trial.
- A block run on the wrong setting still counts as its planned setting,
  and pulls the estimate toward 0. Record it in the trial log. No option
  relabels or drops a block, and an edited plan raises
  [`ScheduleMismatch`](../../reference/errors.md#schedulemismatch).
- A block with no kept sample, from an outage, BAD quality or a trial
  stopped early, refuses the whole analysis (`empty_block`, exit 3). No
  option drops a block today. Planning a shorter window with the same
  seed draws a different schedule, so it does not recover the blocks
  that ran. Report the refusal and the reason.
- Carryover longer than the washout makes early kept samples belong to
  the previous setting. Lengthen the washout.
- Anything else that changes in step with the schedule, such as a manual
  move an operator makes at every switch, is part of the setting.

## Variability: a workaround

A better tuning often shows as less variability around the setpoint,
not a shifted mean, and tsdive estimates a difference in means. As a
workaround, analyze a derived target: the distance from setpoint,
|PV - SP|, written as an archive of its own. A negative B minus A then
says the new tuning holds the loop closer to its setpoint. The demo
outlet has no controller, so this shows the mechanics only:

```pycon
>>> import pandas as pd
>>> import tsdive
>>> ti = pd.read_parquet("data/switchback_demo/ti201.parquet")
>>> dev = ti.assign(value=(ti["value"] - 180.0).abs())
>>> meta = tsdive.TagMeta(identity=tsdive.TagIdentity("demo", "TI201.DEV"),
...                       name="TI-201 distance from setpoint", unit_raw="degC",
...                       sample_rate_s=60.0)
>>> _ = tsdive.write_tag("ti201_dev.parquet", dev, meta)
>>> plan = tsdive.switchback_plan("2024-06-03T00:00:00Z", "2024-06-04T00:00:00Z",
...                               block="PT1H", washout="PT15M", seed=7)
>>> r = tsdive.switchback_analyze(["ti201_dev.parquet"], plan, target="TI201.DEV")
>>> print(f"B - A {r.direct.estimate:+.2f} degC, p {r.direct.p_value:.3f}")
B - A -0.59 degC, p 0.159

```

Take the setpoint from the SP tag when it moves. A rolling standard
deviation over each block's kept samples works the same way. Size the
derived target with its own power readout: it has its own sigma.
