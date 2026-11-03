"""
Deterministic F1--F10 failure trace tagging.
"""

from __future__ import annotations

from typing import Any

from .structured_report_schema import normalize_label


def tag_failure_traces(
    *,
    gold_label: str,
    predicted_label: str,
    evidence_match_result: Any,
    action_eval_result: Any,
    schema_valid: bool,
    repair_failed: bool = False,
    excessive_source_access: bool = False,
) -> dict[str, int]:
    """
    Return deterministic F1--F10 failure flags.

    F1  Label error
    F2  Evidence omission
    F3  Evidence contradiction
    F4  Temporal violation
    F5  Source hallucination
    F6  Causal overclaim
    F7  Action without evidence
    F8  High-risk action violation
    F9  Excessive source access
    F10 Repair failure
    """
    flags = {f"F{i}": 0 for i in range(1, 11)}

    if normalize_label(predicted_label) != normalize_label(gold_label):
        flags["F1"] = 1

    if evidence_match_result.proxy_required_count > 0:
        matched_required = len(set(evidence_match_result.matched_proxy_ids))
        if matched_required < evidence_match_result.proxy_required_count:
            flags["F2"] = 1

    if len(evidence_match_result.value_contradiction_ids) > 0:
        flags["F3"] = 1

    if len(evidence_match_result.temporal_violation_ids) > 0:
        flags["F4"] = 1

    if len(evidence_match_result.invalid_source_or_field_ids) > 0 or not schema_valid:
        flags["F5"] = 1

    if len(evidence_match_result.causal_overclaim_ids) > 0:
        flags["F6"] = 1

    if len(action_eval_result.action_without_evidence_indices) > 0:
        flags["F7"] = 1

    if len(action_eval_result.high_risk_violation_indices) > 0:
        flags["F8"] = 1

    if excessive_source_access:
        flags["F9"] = 1

    if repair_failed:
        flags["F10"] = 1

    return flags