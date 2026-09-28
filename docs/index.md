# tsdive

## Start from your question

| question | run | read |
|---|---|---|
| Is my data trustworthy? | `tsdive profile` | [Ingest a PI, IP.21 or OPC export](guide/howto/historian-export.md), then [reading the profile](guide/output/profile.md) |
| Did the process change? | `tsdive screen`, `spc`, `compare` | [Choose a clean baseline window](guide/howto/baseline-window.md), then [compare's three tables](guide/concepts/compare.md) |
| Did my change work? | `tsdive switchback plan` and `analyze` | [Test a controller tuning](guide/howto/switchback-trial.md) |

New to tsdive? [Getting started](getting-started.md) takes you from
`pip install` to a profile of your own CSV export in about 15 minutes,
with demo data that `tsdive demo` writes.

## What it is

<!-- readme: intro -->

## Install

<!-- readme: install -->

## Contents

- [Getting started](getting-started.md): install, demo data, a first
  profile and screen, then your own export
- User guide: [concepts](guide/concepts/windows-and-time.md) such as
  coverage, censoring and the sampling contract;
  [reading the output](guide/output/profile.md) of every analysis, line
  by line; [how-to guides](guide/howto/historian-export.md) for real
  exports, baselines, many tags, Python, AI assistants and controller
  trials; the [command tour](usage.md) and
  [switchback trials](SWITCHBACK.md)
- Reference: the [CLI](reference/cli.md), the
  [API reference](reference/api/index.md),
  [errors and refusals](reference/errors.md), the
  [glossary](reference/glossary.md), the [archive schema](SCHEMA.md),
  [source adapters](SOURCES.md) and the [MCP server](MCP.md)
- Project: [scope](SCOPE.md), [roadmap](ROADMAP.md),
  [datasets and studies](DATA.md), the
  [evaluation protocol](EVAL.md), [benchmarks](benchmarks.md) and the
  [changelog](changelog.md)
