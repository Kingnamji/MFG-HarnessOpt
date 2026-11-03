"""Factual comparisons with no access to diagnostic labels or target supports."""
from __future__ import annotations
import math
import pandas as pd


def values_equal(first, second) -> bool:
    if first is None or second is None:
        return first is second
    try:
        a, b = float(first), float(second)
        if math.isfinite(a) and math.isfinite(b):
            return math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-6)
        return False
    except (TypeError, ValueError):
        return str(first).strip() == str(second).strip()


def times_equal(first, second) -> bool:
    if first is None or str(first).strip() in {"", "nan", "NaT", "None"}:
        return True  # Time is optional in the report; the named window still must match.
    try:
        return bool(pd.Timestamp(first) == pd.Timestamp(second))
    except (ValueError, TypeError):
        return False
