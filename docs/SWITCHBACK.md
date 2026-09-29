# Switchback trials

This page shows how to plan a trial of two settings on one unit with
`tsdive switchback plan`, how to read it with `tsdive switchback analyze`,
and what the result states.

## The design

A switchback trial runs settings A and B on one unit in alternating time
blocks. `plan` cuts the window into K blocks of `--block` and drops the
last block when K is odd. A seeded balanced draw assigns exactly K/2
blocks to B. `analyze` leaves the first `--washout` of every block out,
so the response to the previous setting can settle.

The plan file records the window, the block length, the washout, the
seed, every block with its setting, and a SHA-256 digest over them. The
same arguments give the same plan and the same digest on every platform.

`analyze` checks the plan before it reads any archive. It recomputes the
digest, the block times, the balance and the settings the seed draws, and
raises `ScheduleMismatch` when one of them differs from the file. `plan`
raises `DesignTooSmall` when K blocks give fewer than 20 balanced
assignments or a smallest two-sided p-value above 0.05, which is every
K below 8.

`analyze` reads the target over the schedule under the sampling contract
`compare` uses and keeps the GOOD numeric samples that sit at least the
washout into their block. It reports:

- the difference in means, B minus A, in the target's unit;
- its randomization p-value over the balanced assignments: all of them
  when there are at most 1000 (K of 12 or fewer), otherwise the schedule
  plus 1000 seeded draws, with p = (1 + #) / 1001;
- the 95% interval that inverts the test under a constant shift. A side
  the interval does not close on prints as `unbounded`.

With `--covariate`, `analyze` also reports the coefficient of the setting
in a least-squares fit on the declared covariates, refitted for every
assignment. A covariate sample joins the target sample with the same
timestamp. Declare the covariates before you look at the result.

`analyze` also tests each covariate's own difference between B and A,
with the same design and washout. A covariate the setting moves, such as
a controller output, carries part of the effect, and the adjusted
estimate can absorb it. When that test gives p < 0.05 the report prints
`covariate <tag> moves with the setting` under the adjusted estimate,
and `--json` sets `moves_with_setting` under `covariate_checks`. Report
the unadjusted estimate then.

| refusal | when |
|---|---|
| `too_few` | fewer than 30 samples kept past the washout |
| `no_spread` | the kept target has MAD 0, or the covariates reproduce it |
| `too_many_covariates` | more than one covariate per 10 kept samples |
| `empty_block` | a block holds no kept sample, so the design the plan states no longer holds |
| `collinear` | the covariates follow the schedule |

`analyze` prints a refusal in place of the numbers, with the counts that
caused it. A refused difference in means exits with status 3; a refused
adjusted estimate beside a difference in means exits 0.

An edited plan file raises `ScheduleMismatch` naming the digest the file
records. Restore the plan file with that digest: the analysis needs the
plan as drawn.

## When to use it

Use a switchback trial when you can switch a setting on the unit and
switch it back: a controller tuning, a setpoint, a recipe parameter. The
response has to settle well inside one block. A change you cannot
reverse, such as a catalyst replacement, gives one period before and one
after. `compare` reads those two periods and its tables name no cause.

## Block length and block count

Make a block several settling times of the target long, so the washout
is a small share of it. A washout of 3 time constants leaves exp(-3), 5%
of a first-order step. With a 5-minute time constant, a 15-minute washout
keeps 45 of every 60 minutes of a 1-hour block.

The block count sets the power. Pass `--history` and `--history-window`
to read it off a period of the target where nothing was switched. `plan`
lays 200 seeded schedules of the same design from the start of the
history window, adds a shift of delta sigma to their B blocks (sigma is
1.4826 MAD of the history), and counts how often the 95% interval
excludes 0. It reports the claim rate at a zero shift, detection at 0.1,
0.25, 0.5 and 1 sigma, and the smallest of those shifts detected at 0.8
or more. For a history window shorter than the schedule, `plan` reports
`history_short` in place of the readout and still writes the plan. The
readout reads the target alone. It states the power of the difference in
means, and an adjusted estimate can detect smaller shifts.

## Assumptions

The result is the difference between settings A and B under the declared
random schedule, by intended assignment. It holds if the schedule was
followed and carryover ended within the washout. A block run on the
wrong setting stays in the analysis under the setting the plan gave it.

## Transcript

`examples/switchback/make_trial.py` writes three tags of a synthetic unit
to `data/switchback_demo/`: two days of history and one day on which the
plant follows the plan below. Setting B raises the outlet temperature by
0.25 degC through a 5-minute lag, and a drifting feed flow moves it too.
Without a clone, `tsdive demo data` writes the same three archives. The
[controller trial how-to](guide/howto/switchback-trial.md) walks through
sizing and reading a trial for a process engineer.

```tsdive
tsdive switchback plan \
    --window "2024-06-03T00:00:00Z/2024-06-04T00:00:00Z" \
    --block PT1H --washout PT15M --seed 7 \
    --history data/switchback_demo/ti201.parquet \
    --history-window "2024-06-02T00:00:00Z/2024-06-03T00:00:00Z" \
    -o data/switchback_demo/plan.json
tsdive switchback analyze data/switchback_demo/*.parquet \
    --plan data/switchback_demo/plan.json --target TI201.PV \
    --covariate FI200.PV --covariate TT001.PV
```

On the history, 24 blocks detect a 1 sigma shift of the raw target in
62% of schedules, because the feed drift makes sigma 0.45 degC. The
difference in means carries that drift, and its interval holds 0. The
adjusted estimate takes the feed flow and the ambient temperature out
and its interval, 0.22 to 0.34 degC, holds the 0.25 degC the script
added. `--json` prints the same fields as one object, and the plan file
is the object `plan --json` prints after its `result_kind` and
`tsdive_version`.

## Python and `tsdive run`

`tsdive.switchback_plan(start, end, block, washout, seed, history=...,
history_window=...)` returns the `SwitchbackPlan` that `write_json` and
`SwitchbackPlan.read_json` store and load.
`tsdive.switchback_analyze(archives, plan, target=..., covariates=...)`
returns a `SwitchbackAnalysis` with `render()`, `to_dict()` and `frame`.
In a `tsdive run` plan, list `switchback` under `steps` and give it
`plan`, `target` and `covariate` under `[options.switchback]`. A relative
`plan` path resolves against the plan file, as the archive globs do.

## Study numbers

The switchback study adds a known shift to randomly assigned blocks of
records where nothing was changed: 3W, TEP, the Turbine Upgrade pairs
and SKAB.

- On fresh assignments the claim rate at a zero shift is 3.9% to 5.3%
  per bed. A first-half against second-half split of the same 3W
  records claims a shift on 65.2% of tags.
- On the enumerated 3W designs no (record, target) rejects more than
  0.0286 of its assignments, against a bound of 0.05. The inverted
  interval meets its coverage bar on every bed, and a washout of 3 time
  constants removes most of the carryover bias.
- Detection of a 0.25 sigma shift reaches 0.922 with 212 and 414
  one-day blocks on the turbine pairs, and 0.113 with 16 blocks of 15
  minutes on a 4-hour 3W record.

Report: [examples/studies/switchback/REPORT.md](../examples/studies/switchback/REPORT.md).
