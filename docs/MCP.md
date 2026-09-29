# MCP server

`tsdive-mcp` serves six tsdive analyses to an MCP client over stdio.
This page lists the tools, the shape of every answer, and how to launch
the server.

## Install

The server needs the `mcp` extra. A base install does not carry it.
Install tsdive with the extra from git, or from the release wheel:

```console
pip install "tsdive[mcp] @ git+https://github.com/NorthernLightx/tsdive.git"
pip install "tsdive[mcp] @ https://github.com/NorthernLightx/tsdive/releases/download/vX.Y.Z/tsdive-X.Y.Z-py3-none-any.whl"
```

From a clone, `pip install -e ".[mcp]"` installs it. Without the extra,
`tsdive-mcp` writes one line to stderr and exits 2:

```console
error: the tsdive MCP server needs the mcp package (>=2.1); install tsdive with the mcp extra: pip install "tsdive[mcp] @ git+https://github.com/NorthernLightx/tsdive.git", or the release wheel with it, pip install "tsdive[mcp] @ https://github.com/NorthernLightx/tsdive/releases/download/vX.Y.Z/tsdive-X.Y.Z-py3-none-any.whl"
```

## Launch

```json
{
  "mcpServers": {
    "tsdive": {
      "command": "tsdive-mcp"
    }
  }
}
```

stdio is the only transport. The server reads local parquet archives and
opens no network connection. It writes nothing, and every tool carries
`readOnlyHint: true`.

## Tools

Each tool calls the function behind `tsdive <command> --json`, so the
same arguments give the same fields as the command line.

| Tool | Required | Optional | Evidence carries |
|---|---|---|---|
| `profile` | `archive` | `window`, `basis`, `stepped`, `tz`, `flatline` | sampling contract, unit resolution, coverage and gap classes, clipping, timestamp audit, quality counts, statistics, flatline verdict |
| `segment` | `archive` | `window`, `penalty`, `min_size`, `basis`, `stepped` | breakpoints, and one row per segment with `n`, `median` and `mad` |
| `screen` | `archive`, `baseline`, `window` | `method`, `k`, `mode`, `basis`, `stepped` | center, scale, limits, `n_screened`, `n_flagged`, and the flagged timestamps |
| `spc` | `archive`, `baseline`, `window` | `basis`, `stepped` | individuals limits, and the hits of `BEYOND_3SIGMA`, `RUN_9_SAMESIDE` and `TREND_6` |
| `compare` | `archives`, `before`, `after` | `top`, `rate_s` | per-tag change rows, the pairwise correlation table, the joint MSPC structure |
| `switchback_analyze` | `archives`, `plan`, `target` | `covariates` | the verified plan digest, the B minus A estimate with its p-value and 95% interval, kept samples per block, the adjusted estimate |

`archive` is the path to one single-tag parquet archive. `compare` and
`switchback_analyze` take a list of them in `archives`. `plan` is the
path of a file `tsdive switchback plan` wrote.

Windows are ISO 8601 in UTC: START/END written
`2024-03-01T00:00:00Z/2024-03-01T01:00:00Z`, START/PT1H, PT1H/END, or a
date for one whole UTC day. `screen` and `spc` take a
`baseline` window and a `window` to monitor. `compare` takes `before`
and `after`. An option left out takes the default the command line
declares, so `k` is 3.0, `min_size` is 10 and `top` is 10.

## Result union

Every tool returns one JSON object carrying `result_kind`.

`"evidence"` holds the analysis, keyed as `--json` prints it. A
`profile` answer, trimmed to four of its ten blocks:

```json
{
  "result_kind": "evidence",
  "tag": "demo:FIC101.PV",
  "window": {"start": "2024-03-30T20:00:00+00:00", "end": "2024-03-30T23:00:00+00:00", "duration_s": 10800.0},
  "units": {"raw": "m3/h", "canonical": "cubic meters per hour", "resolved": true},
  "coverage": {"coverage": 1.0, "n_gaps": 0, "longest_gap_s": null}
}
```

`"refusal"` holds a check that has no answer on this data:

```json
{
  "result_kind": "refusal",
  "error_type": "InsufficientQuality",
  "cause": "tag demo:FIC101.PV: window is censored (clipped fraction 0.935); a clipped window may never serve as a baseline"
}
```

`isError` stays false on a refusal, so the call succeeds and the client
reads the reason from `error_type` and `cause`. `tsdive <command> --json`
prints the same object on stdout for a refusal and exits with status 3. `error_type` names a
class from `tsdive.errors`: `SchemaError`, `InsufficientQuality`,
`IncomparableSamplingError`, `NonMonotonicIndex`,
`UnresolvedUnitError`, `IncomparableUnitsError`, `RegimeTooSparse`,
`ZeroSpreadBaseline`, `PopulationTooSparse`, `GroupLeakage`, `MspcAlignmentError`,
`DesignTooSmall`, `ScheduleMismatch` or `NarratorUnavailable`. `docs/SCOPE.md` states which check raises which
class.

## Tool errors

A tool error says the call was built wrong. A malformed window, a
rejected option value, an unreadable path, or a baseline overlapping the
monitored window raises instead of returning. The client receives
`isError: true` and the message the command line prints:

```console
Error executing tool profile: window must be <START>/<END>, <START>/<DURATION>, <DURATION>/<END> or <DATE> in ISO 8601 UTC
```
