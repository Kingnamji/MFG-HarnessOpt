"""
Structured diagnostic report schema utilities.

This module validates the structured diagnostic report returned by each harness.
It performs only deterministic schema and whitelist checks. Semantic checks such
as evidence matching, causal overclaim, and action precondition validation are
handled by separate evaluator modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


REQUIRED_REPORT_KEYS = {
    "predicted_label",
    "confidence",
    "evidence",
    "next_steps",
    "limitations",
}

REQUIRED_EVIDENCE_KEYS = {
    "evidence_id",
    "source",
    "field",
    "statistic",
    "direction",
    "value",
    "window",
    "supports",
    "role",
}

REQUIRED_ACTION_KEYS = {
    "action_type",
    "target",
    "rationale_evidence_ids",
}

ALLOWED_CONFIDENCE_VALUES = {"low", "medium", "high"}


@dataclass(frozen=True)
class SchemaValidationResult:
    valid: bool
    errors: list[str]


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def normalize_source(value: Any) -> str:
    return normalize_text(value).replace(" ", "_").replace("-", "_")


def normalize_action_type(value: Any) -> str:
    return normalize_text(value).replace(" ", "_").replace("-", "_")


def normalize_label(value: Any) -> str:
    """
    Normalize labels while preserving the canonical labels used in the paper.

    Examples
    --------
    "twf" -> "TWF"
    "No Failure" -> "No Failure"
    "comp2" -> "comp2"
    """
    raw = str(value).strip()
    low = raw.lower().replace("_", " ")

    if low in {"no failure", "normal", "none"}:
        return "No Failure"

    if low in {"twf", "hdf", "pwf", "osf", "rnf"}:
        return low.upper()

    if low in {"comp1", "comp2", "comp3", "comp4"}:
        return low

    return raw


def canonicalize_field(
    field: Any,
    alias_map: dict[str, list[str]] | None = None,
) -> str:
    """
    Canonicalize evidence field names.

    alias_map example:
    {
        "mechanical_power_w": ["power", "power_w", "computed_power"],
        "rotational_speed": ["rpm", "rotation", "speed"]
    }
    """
    normalized = normalize_text(field).replace(" ", "_").replace("-", "_")

    if not alias_map:
        return normalized

    for canonical, aliases in alias_map.items():
        canonical_norm = normalize_text(canonical).replace(" ", "_").replace("-", "_")

        if normalized == canonical_norm:
            return canonical_norm

        for alias in aliases:
            alias_norm = normalize_text(alias).replace(" ", "_").replace("-", "_")
            if normalized == alias_norm:
                return canonical_norm

    return normalized


def ensure_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def validate_report_schema(
    report: dict[str, Any],
    candidate_labels: list[str],
    allowed_sources: dict[str, list[str]] | None = None,
    allowed_action_types: list[str] | None = None,
    alias_map: dict[str, list[str]] | None = None,
) -> SchemaValidationResult:
    """
    Validate the diagnostic report structure and whitelist membership.

    This function does not decide whether an evidence claim is true. It only
    checks whether the output follows the required schema and uses allowed
    labels, sources, fields, and action types.
    """
    errors: list[str] = []

    if not isinstance(report, dict):
        return SchemaValidationResult(False, ["report is not a dictionary"])

    missing_top = REQUIRED_REPORT_KEYS - set(report.keys())
    if missing_top:
        errors.append(f"missing top-level keys: {sorted(missing_top)}")

    predicted_label = normalize_label(report.get("predicted_label", ""))
    allowed_labels = {normalize_label(label) for label in candidate_labels}

    if predicted_label not in allowed_labels:
        errors.append(f"invalid predicted_label: {predicted_label}")

    confidence = normalize_text(report.get("confidence", ""))
    if confidence and confidence not in ALLOWED_CONFIDENCE_VALUES:
        errors.append(f"invalid confidence: {confidence}")

    evidence_items = report.get("evidence", [])
    if not isinstance(evidence_items, list):
        errors.append("evidence is not a list")
        evidence_items = []

    allowed_sources = allowed_sources or {}

    for idx, evidence in enumerate(evidence_items):
        if not isinstance(evidence, dict):
            errors.append(f"evidence[{idx}] is not a dictionary")
            continue

        missing_evidence = REQUIRED_EVIDENCE_KEYS - set(evidence.keys())
        if missing_evidence:
            errors.append(f"evidence[{idx}] missing keys: {sorted(missing_evidence)}")

        support_value = evidence.get("supports", "")
        support = str(support_value).strip()
        if not isinstance(support_value, str) or (normalize_label(support) not in allowed_labels and support != "context"):
            errors.append(f"evidence[{idx}] supports must be one candidate label or context")

        source = normalize_source(evidence.get("source", ""))
        field = canonicalize_field(evidence.get("field", ""), alias_map)

        if allowed_sources:
            if source not in allowed_sources:
                errors.append(f"evidence[{idx}] invalid source: {source}")
            else:
                allowed_fields = {
                    canonicalize_field(f, alias_map)
                    for f in allowed_sources[source]
                }
                if field not in allowed_fields:
                    errors.append(
                        f"evidence[{idx}] invalid field '{field}' for source '{source}'"
                    )

    actions = report.get("next_steps", [])
    if not isinstance(actions, list):
        errors.append("next_steps is not a list")
        actions = []

    allowed_action_set = {
        normalize_action_type(action_type)
        for action_type in (allowed_action_types or [])
    }

    for idx, action in enumerate(actions):
        if not isinstance(action, dict):
            errors.append(f"next_steps[{idx}] is not a dictionary")
            continue

        missing_action = REQUIRED_ACTION_KEYS - set(action.keys())
        if missing_action:
            errors.append(f"next_steps[{idx}] missing keys: {sorted(missing_action)}")

        action_type = normalize_action_type(action.get("action_type", ""))
        if allowed_action_set and action_type not in allowed_action_set:
            errors.append(f"next_steps[{idx}] invalid action_type: {action_type}")

        rationale_ids = action.get("rationale_evidence_ids", [])
        if not isinstance(rationale_ids, list):
            errors.append(
                f"next_steps[{idx}] rationale_evidence_ids is not a list"
            )

    limitations = report.get("limitations", [])
    if not isinstance(limitations, list):
        errors.append("limitations is not a list")

    return SchemaValidationResult(valid=(len(errors) == 0), errors=errors)
    
