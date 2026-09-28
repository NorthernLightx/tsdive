# Connect an AI assistant

`tsdive-mcp` serves six analyses to an AI assistant that speaks the
Model Context Protocol (MCP), such as Claude Desktop or Claude Code. The
assistant can then profile, segment, screen, chart and compare your
archives, and read a switchback trial, while you ask in plain words.
This guide installs the server and wires it into a client.

## Install the extra

The server needs the `mcp` extra. Install the wheel with it:

```console
uv pip install "tsdive[mcp] @ https://github.com/NorthernLightx/tsdive/releases/download/vX.Y.Z/tsdive-X.Y.Z-py3-none-any.whl"
```

With plain pip the command is the same after `pip install`. From a
clone, `pip install -e ".[mcp]"` installs the extra.

## Find the server's path

A desktop client does not know your virtual environment, so give it the
absolute path of the `tsdive-mcp` executable. With the environment
active, `where tsdive-mcp` on Windows or `which tsdive-mcp` on Linux and
macOS prints it. It sits in the environment's `Scripts` folder on
Windows and its `bin` folder elsewhere.

## Wire it into the client

Claude Desktop reads `claude_desktop_config.json`, which its settings
open under Developer. Add the server under `mcpServers`, with your path:

```json
{
  "mcpServers": {
    "tsdive": {
      "command": "C:\\path\\to\\project\\.venv\\Scripts\\tsdive-mcp.exe"
    }
  }
}
```

On Linux and macOS the command is a path such as
`/home/me/project/.venv/bin/tsdive-mcp`. Restart the client, and the
six tools appear in its tool list. Claude Code adds the same server
with `claude mcp add tsdive /absolute/path/to/tsdive-mcp`.

The server speaks stdio only, reads local parquet archives, and opens no
network connection. It writes nothing, and every tool is marked
read-only.

## What the assistant receives

Each tool returns the document `tsdive <command> --json` prints, with a
`result_kind` of `evidence`. Ask "screen the demo flow from 01:00
against 20:00 to 01:00", and the assistant calls:

```pycon
>>> from tsdive.mcp_server import screen
>>> answer = screen(archive="data/demo/fic101_demo.parquet",
...                 baseline="2024-03-30T20:00:00Z/2024-03-31T01:00:00Z",
...                 window="2024-03-31T01:00:00Z/2024-03-31T06:00:00Z")
>>> answer["result_kind"], answer["n_flagged"], answer["n_screened"]
('evidence', 29, 300)

```

A question the data cannot answer returns a refusal, not a tool error,
so the assistant can explain it instead of retrying:

```pycon
>>> answer = screen(archive="data/demo/fic101_demo.parquet",
...                 baseline="2024-03-31T00:00:00Z/2024-03-31T03:00:00Z",
...                 window="2024-03-31T03:00:00Z/2024-03-31T06:00:00Z")
>>> answer["result_kind"], answer["error_type"]
('refusal', 'InsufficientQuality')

```

A call built wrong, such as a malformed window, is a tool error with
the message the command line prints.

## What to ask

- "Profile FIC101 over last night and tell me whether the data is
  trustworthy."
- "Which tags of unit 3 changed between last week and yesterday?"
- "Read the switchback trial in plan.json for TI201.PV with FI200.PV as
  a covariate."

Give the assistant archive paths it can read. It works on archives, so
ingest exports first with [the export guide](historian-export.md).
[MCP server](../../MCP.md) lists every tool, its arguments and the shape
of every answer.
