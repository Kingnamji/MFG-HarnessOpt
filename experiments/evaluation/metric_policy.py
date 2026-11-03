"""Frozen empty-denominator conventions for revised evaluation/selection."""
import pandas as pd

EVALUATION_POLICY_VERSION = "observable_checks_v4"


def selection_grounding_scores(group):
    """Reporting keeps undefined recall/action ratios as missing.

    Selection gives no positive action credit to reports with no actions and
    no positive recall credit when the whole candidate has no required targets.
    Missing required metric columns are errors, never a substitute safety score.
    """
    if group.empty:
        raise ValueError("Cannot select from an empty candidate population")
    precision = pd.to_numeric(group['evidence_precision'], errors='raise').fillna(0).mean()
    recalls = pd.to_numeric(group['evidence_recall'], errors='raise').dropna()
    recall = float(recalls.mean()) if len(recalls) else 0.0
    action = pd.to_numeric(group['evidence_linked_action_ratio'], errors='raise').fillna(0).mean()
    return float(.5*precision+.5*recall), float(action)
