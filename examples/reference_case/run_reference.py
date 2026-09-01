"""Reference case stubs: two refusal demos, runnable day one.

Each demo renders a deterministic report, asserts the typed reason code
appears verbatim, and compares against the committed file under
outputs/ byte-for-byte.

    uv run python examples/reference_case/run_reference.py            # verify
    uv run python examples/reference_case/run_reference.py --update   # regenerate

Answerable-question and negative-control cases arrive with stages 2-4
(see docs/ROADMAP.md).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

OUTPUTS = Path(__file__).parent / "outputs"

CASE_UNIT_MISMATCH = "unit_mismatch_refusal.txt"
CASE_INSUFFICIENT_QUALITY = "insufficient_quality_refusal.txt"


def render_unit_mismatch() -> str:
    from tsdive.errors import IncomparableUnitsError
    from tsdive.store.units import resolve_unit

    lines = [
        "reference case 1: unit-mismatch refusal",
        "",
        "tags:",
        "  plant1:FIC101.AIR_FLOW  unit_raw=Nm3/h "
        f"(reference_condition={resolve_unit('Nm3/h').reference_condition})",
        "  plant1:FIC101.N2_FLOW   unit_raw=m3/h "
        f"(reference_condition={resolve_unit('m3/h').reference_condition})",
        "requested: cross-tag comparison of air flow vs nitrogen flow",
        "declared reference state for Nm3/h: none",
        "",
        "result: REFUSED",
    ]
    try:
        from tsdive.store.units import check_comparable

        check_comparable("Nm3/h", "m3/h")
        raise AssertionError("expected refusal")
    except IncomparableUnitsError as e:
        lines.append(f"[{type(e).__name__}] {e}")
    lines.append("reason code present verbatim: IncomparableUnitsError")
    return "\n".join(lines) + "\n"


def render_insufficient_quality() -> str:
    import pandas as pd

    from tsdive.errors import InsufficientQuality
    from tsdive.store.averaging import weighted_mean
    from tsdive.store.quality import annotate_severity, null_digital_state_values
    from tsdive.store.sampling_contract import (
        CalculationBasis,
        RetrievalMode,
        SamplingContract,
    )

    stamps = [pd.Timestamp(f"2024-03-01 00:00:0{i}+00:00") for i in range(4)]
    df = pd.DataFrame(
        {
            "timestamp": stamps,
            "value": [100.0, 100.0, 12.5, 11.0],
            "quality": ["OVER RANGE", 257, "COMM FAILURE", "BAD"],
        }
    )
    annotated = null_digital_state_values(annotate_severity(df))
    contract = SamplingContract(CalculationBasis.EVENT_WEIGHTED, RetrievalMode.RECORDED)

    counts_line = ", ".join(
        f"{s}={n}" for s, n in annotated["severity"].value_counts().sort_index().items()
    )
    lines = [
        "reference case 2: insufficient-quality refusal",
        "",
        "window: plant1:TIC101.PV [2024-03-01T00:00:00+00:00 -> 2024-03-01T00:00:06+00:00]",
        f"samples: {len(df)}; severity counts: {counts_line}",
        "GOOD samples remaining after masking: 0",
        "requested: event-weighted mean over GOOD samples only",
        "",
        "result: REFUSED",
    ]
    try:
        weighted_mean(annotated, contract)
        raise AssertionError("expected refusal")
    except InsufficientQuality as e:
        lines.append(f"[{type(e).__name__}] {e}")
    lines.append("reason code present verbatim: InsufficientQuality")
    return "\n".join(lines) + "\n"


CASES = {
    CASE_UNIT_MISMATCH: (render_unit_mismatch, "IncomparableUnitsError"),
    CASE_INSUFFICIENT_QUALITY: (render_insufficient_quality, "InsufficientQuality"),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update", action="store_true", help="rewrite committed outputs")
    args = parser.parse_args(argv)

    OUTPUTS.mkdir(exist_ok=True)
    failures = 0
    for name, (render, reason_code) in CASES.items():
        rendered = render()
        if reason_code not in rendered:
            print(f"FAIL {name}: reason code {reason_code} missing from rendered output")
            failures += 1
            continue
        path = OUTPUTS / name
        if args.update or not path.exists():
            path.write_text(rendered, encoding="utf-8", newline="\n")
            print(f"wrote {path}")
            continue
        committed = path.read_text(encoding="utf-8")
        if committed == rendered:
            print(f"OK   {name}: byte-identical")
        else:
            print(f"FAIL {name}: differs from committed output; rerun with --update")
            failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
