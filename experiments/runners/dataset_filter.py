from __future__ import annotations

from typing import Iterable


VALID_DATASETS = {"AI4I", "PdM"}


def parse_datasets_arg(datasets: str | None) -> set[str]:
    """
    Parse CLI dataset filter.

    Accepted:
    - None or empty: {"AI4I", "PdM"}
    - "AI4I"
    - "PdM"
    - "AI4I,PdM"
    - "Azure" or "Azure PdM" maps to "PdM"
    """
    if datasets is None or not str(datasets).strip():
        return {"AI4I", "PdM"}

    parsed: set[str] = set()

    for item in str(datasets).split(","):
        value = item.strip().lower()

        if value in {"ai4i", "ai4i 2020"}:
            parsed.add("AI4I")
        elif value in {"pdm", "azure", "azure pdm", "microsoft azure predictive maintenance"}:
            parsed.add("PdM")
        else:
            raise ValueError(
                f"Unsupported dataset filter: {item}. "
                "Use AI4I, PdM, or AI4I,PdM."
            )

    if not parsed:
        raise ValueError("Dataset filter cannot be empty.")

    return parsed


def validate_requested_datasets(requested: Iterable[str]) -> set[str]:
    requested = set(requested)
    invalid = requested - VALID_DATASETS
    if invalid:
        raise ValueError(f"Invalid datasets: {sorted(invalid)}")
    return requested