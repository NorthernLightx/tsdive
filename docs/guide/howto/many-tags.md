# Check many tags at once

`tsdive run` walks one plan over many archives and writes one
[evidence ledger](../../reference/glossary.md#evidence-ledger): a row
per tag, every profile, every finding, every refusal and every error,
as text, JSON and a static HTML page. This guide writes a plan for the two demo tags and reads
what it produces.

## Write the plan

A plan is a TOML file. Archive paths and globs are read relative to the
plan's own folder:

```toml file=unit.toml
archives = ["data/demo/*.parquet"]
baseline = "2024-03-30T20:00:00Z/2024-03-30T23:00:00Z"
window   = "2024-03-31T04:00:00Z/2024-03-31T06:00:00Z"
before   = "2024-03-30T20:00:00Z/2024-03-30T23:00:00Z"
after    = "2024-03-31T04:00:00Z/2024-03-31T06:00:00Z"
steps    = ["profile", "screen", "spc", "mspc", "compare"]

[options.profile]
flatline = true
```

| key | holds |
|---|---|
| `archives` | paths or globs of the archives to read |
| `window` | the window `profile`, `segment`, `screen`, `spc` and `mspc` read |
| `baseline` | the baseline `screen`, `spc` and `mspc` judge the window against |
| `before`, `after` | the two periods of `compare` |
| `steps` | analyses to run, any of `profile`, `segment`, `screen`, `spc`, `mspc`, `compare`, `switchback` |
| `[options.<step>]` | that command's flags, spelled as keys without the dashes: `flatline = true`, `tz = ["Europe/London"]`, `rate_s = 60` |

The steps run in pipeline order, whatever order the list gives. `mspc`
and `compare` read every archive at once; the others run once per
archive. An unknown key or step stops the run before anything is
written.

## Run it

```tsdive
tsdive run unit.toml -o unit-run
```

The headline counts the archives, steps, profiles, findings and
refusals. The baseline here avoids the outage at 23:00 and the clipped
stretch at 02:00, so no step is refused. Move the baseline to 20:00 to
01:00 and `mspc` becomes a refusal row with the outage as its reason,
while the other steps still run.

## Read the ledger

`ledger.txt` opens with the text the command printed. A `TAGS` table
follows, one row per archive, then the full text of every profile and
every finding, in step order:

```text show=unit-run/ledger.txt lines=40
```

The `TAGS` row reads the archive's profile: coverage, the GOOD share of
its samples, the censoring verdict, the gap count, the longest gap, the
flatline verdict (`not run` without `flatline = true`) and the steps
refused for that tag. A tag the `profile` step did not read shows
`n/a`.

`ledger.json` holds the same content for a script:

| key | holds |
|---|---|
| `title` | the plan's file name |
| `tsdive_version` | the version that wrote the ledger |
| `tags` | one object per archive: `tag`, `coverage`, `good_share`, `censored`, `gaps`, `longest_gap_s`, `flatline`, `refused`. The profile keys are `null` for a tag the `profile` step did not read |
| `profiles` | one rendered profile report per archive |
| `findings` | one object per result: `step`, `tags`, the rendered `text`, and `data`, the document the command prints under `--json` |
| `refusals` | one object per step that raised a typed error: `step`, `tags`, `error_type`, `cause` |
| `errors` | one object per step that raised any other error, in the same keys: a rejected option, overlapping windows, a file the OS cannot read |
| `benchmark_rows` | empty for a plan run |

A refusal row is a result about the data of that tag. An error row
names a plan the step could not run, the input that exits 2 on the
command line.

`report.html` is a static page with the tag table, every profile, a
figure per archive, the refusal log and the error log. It needs no
server: open it in a browser, or attach it to an e-mail.

## Exit status

`tsdive run` exits 0 when the ledger holds at least one profile or
finding, refusal and error rows included. It exits 2 when the plan
cannot be read or no step produced a result.

With `--strict` the rows decide the status, as they would for the
single commands:

| status | when |
|---|---|
| 0 | no refusal and no error row |
| 2 | at least one error row |
| 3 | at least one refusal row and no error row |

A scheduled job passes `--strict`, or reads the `refusals` and `errors`
lists of `ledger.json`.

## What to do

- Put every tag of a unit in one plan, so `compare` and `mspc` see
  them together.
- Keep the plan next to the ledger it wrote: the pair records what was
  asked and what was answered.
- Choose the baseline with [the baseline guide](baseline-window.md)
  before running a plan over many tags; a baseline that suits one tag
  may clip on another.
