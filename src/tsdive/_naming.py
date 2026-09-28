"""How an error message names an option: the CLI flag or the Python keyword.

The same functions run under ``tsdive <command>`` and under a Python
call, so a message that tells the caller what to pass has two spellings,
``--rate-s`` and ``rate_s=``. The CLI marks the commands it runs with
``cli_names``; everywhere else, the MCP server included, messages name
the keyword.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from contextvars import ContextVar

_CLI: ContextVar[bool] = ContextVar("tsdive_cli_names", default=False)


def cli_active() -> bool:
    """True while the CLI runs a command."""
    return _CLI.get()


def argname(keyword: str, flag: str) -> str:
    """``flag`` while the CLI runs a command, else ``keyword``."""
    return flag if _CLI.get() else keyword


def option(keyword: str, flag: str, value: str | None = None) -> str:
    """How to pass an option: ``--flag VALUE`` in the CLI, ``keyword=VALUE`` elsewhere.

    Examples:
        >>> from tsdive._naming import cli_names, option
        >>> option("rate_s", "--rate-s")
        'rate_s='
        >>> option("tz", "--tz", "<IANA zone>")
        'tz=<IANA zone>'
        >>> with cli_names():
        ...     option("rate_s", "--rate-s"), option("tz", "--tz", "<IANA zone>")
        ('--rate-s', '--tz <IANA zone>')
    """
    if _CLI.get():
        return flag if value is None else f"{flag} {value}"
    return f"{keyword}=" if value is None else f"{keyword}={value}"


@contextlib.contextmanager
def cli_names() -> Iterator[None]:
    """Name CLI flags in every message raised inside the block."""
    token = _CLI.set(True)
    try:
        yield
    finally:
        _CLI.reset(token)
