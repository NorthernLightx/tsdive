# Check many tags at once

`tsdive run` walks one plan over many archives and writes one
[evidence ledger](../../reference/glossary.md#evidence-ledger): every
profile, every finding and every refusal, as text, JSON and a static
HTML page. This guide writes a plan for the two demo tags and reads
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

`ledger.txt` holds the full text of every result, in step order:

```text show=unit-run/ledger.txt lines=40
```

`ledger.json` holds the same content for a script:

| key | holds |
|---|---|
| `title` | the plan's file name |
| `tsdive_version` | the version that wrote the ledger |
| `profiles` | one rendered profile report per archive |
| `findings` | one object per result: `step`, `tags` and the rendered `text` |
| `refusals` | one line per refused step: `[ErrorName] step tags: message` |
| `benchmark_rows` | empty for a plan run |

The findings carry the rendered text, not the fields of `--json`. For
the fields, call the command with `--json` or use the Python API.

`report.html` is a static page with every profile, a figure per archive
and the refusal log. It needs no server: open it in a browser, or attach
it to an e-mail.

## Exit status

`tsdive run` exits 0 when the ledger holds at least one profile or
finding, refusals included as rows. It exits 2 when the plan cannot be
read or no step produced a result. A script that needs to know about
refusals reads the `refusals` list of `ledger.json`.

## What to do

- Put every tag of a unit in one plan, so `compare` and `mspc` see
  them together.
- Keep the plan next to the ledger it wrote: the pair records what was
  asked and what was answered.
- Choose the baseline with [the baseline guide](baseline-window.md)
  before running a plan over many tags; a baseline that suits one tag
  may clip on another.
