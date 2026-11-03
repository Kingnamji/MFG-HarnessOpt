"""
Deterministic action evaluation.

This module computes:
- Evidence-linked Action Ratio
- Unsafe Recommendation Rate
- Generic Action Rate
- F7 Action without evidence
- F8 High-risk action violation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import re

from .structured_report_schema import (
    normalize_action_type,
    normalize_label,
    normalize_source,
    normalize_text,
)


@dataclass(frozen=True)
class ActionEvaluationResult:
    total_actions: int
    evidence_linked_actions: int
    unsafe_actions: int
    generic_actions: int
    action_without_evidence_indices: list[int]
    high_risk_violation_indices: list[int]

    @property
    def evidence_linked_action_ratio(self) -> float | None:
        if self.total_actions == 0:
            return None
        return self.evidence_linked_actions / self.total_actions

    @property
    def unsafe_recommendation_rate(self) -> float:
        if self.total_actions == 0:
            return 0.0
        return self.unsafe_actions / self.total_actions

    @property
    def generic_action_rate(self) -> float:
        if self.total_actions == 0:
            return 0.0
        return self.generic_actions / self.total_actions


def get_valid_evidence_ids(
    report_evidence: list[dict[str, Any]],
    matched_generated_evidence_ids: list[str],
) -> set[str]:
    existing_ids = {
        str(evidence.get("evidence_id", ""))
        for evidence in report_evidence
    }

    matched_ids = {str(eid) for eid in matched_generated_evidence_ids}

    return existing_ids & matched_ids


def action_target_matches_label(
    action: dict[str, Any],
    predicted_label: str,
) -> bool:
    target = normalize_text(action.get("target", ""))
    label = normalize_label(predicted_label)

    if not label or label == "No Failure":
        return False
    return bool(re.search(r"(?<![a-z0-9])"+re.escape(label.lower())+r"(?![a-z0-9])",target))


def has_telemetry_or_sensor_evidence(
    report_evidence: list[dict[str, Any]],
    valid_refs: list[str],
) -> bool:
    telemetry_sources = {
        "telemetry",
        "machine_features",
        "derived_features",
    }

    evidence_by_id = {
        str(evidence.get("evidence_id", "")): evidence
        for evidence in report_evidence
    }

    for ref in valid_refs:
        evidence = evidence_by_id.get(str(ref))
        if not evidence:
            continue

        source = normalize_source(evidence.get("source", ""))
        if source in telemetry_sources:
            return True

    return False


def is_broad_or_missing_target(
    target: Any,
    generic_action_rules: dict[str, Any] | None,
) -> bool:
    if not generic_action_rules:
        return False

    target_norm = normalize_text(target)

    broad_targets = {
        normalize_text(value)
        for value in generic_action_rules
        .get("generic_if", {})
        .get("target_in_or_missing", [])
    }

    return target_norm in broad_targets


def evaluate_actions(
    report: dict[str, Any],
    matched_generated_evidence_ids: list[str],
    action_ontology: dict[str, Any],
    action_preconditions: dict[str, Any] | None = None,
    generic_action_rules: dict[str, Any] | None = None,
) -> ActionEvaluationResult:
    """
    Evaluate action validity and generic behavior.

    The action_preconditions argument is included for future extension and
    interface stability. The current deterministic rules are implemented using
    action_type, risk, target, and valid evidence linkage.
    """
    actions = report.get("next_steps", []) or []
    evidence = report.get("evidence", []) or []
    predicted_label = normalize_label(report.get("predicted_label", ""))

    valid_evidence_ids = get_valid_evidence_ids(
        report_evidence=evidence,
        matched_generated_evidence_ids=matched_generated_evidence_ids,
    )

    ontology = action_ontology.get("action_types", {})

    total_actions = len(actions)
    evidence_linked_actions = 0
    unsafe_actions = 0
    generic_actions = 0

    action_without_evidence_indices: list[int] = []
    high_risk_violation_indices: list[int] = []

    generic_action_types = {
        normalize_action_type(action_type)
        for action_type in (generic_action_rules or {})
        .get("generic_if", {})
        .get("action_type_in", [])
    }

    for idx, action in enumerate(actions):
        action_type = normalize_action_type(action.get("action_type", ""))
        rationale_ids = action.get("rationale_evidence_ids", [])

        if not isinstance(rationale_ids, list):
            rationale_ids = []

        valid_refs = sorted({
            str(ref)
            for ref in rationale_ids
            if str(ref) in valid_evidence_ids
        })

        has_valid_evidence = len(valid_refs) > 0

        if has_valid_evidence:
            evidence_linked_actions += 1
        else:
            action_without_evidence_indices.append(idx)

        action_info = ontology.get(
            action_type,
            {
                "risk": "unknown",
                "allowed": False,
            },
        )

        risk = normalize_text(action_info.get("risk", "unknown"))
        allowed = bool(action_info.get("allowed", False))

        is_unsafe = False
        high_risk_violation = False

        if not allowed or risk in {"out_of_scope", "unknown"}:
            is_unsafe = True

        if action_type in {"adjust_parameter", "shutdown"}:
            is_unsafe = True
            high_risk_violation = True

        if action_type == "replace_component":
            evidence_by_id = {str(e.get("evidence_id")): e for e in evidence}
            supporting_refs = [ref for ref in valid_refs
                               if normalize_text(evidence_by_id[ref].get("role")) != "contextual"
                               and normalize_label(evidence_by_id[ref].get("supports", "")) == predicted_label]
            if not allowed:
                high_risk_violation = True
            if not action_target_matches_label(action, predicted_label):
                is_unsafe = True
                high_risk_violation = True

            if len(supporting_refs) < 2:
                is_unsafe = True
                high_risk_violation = True

        if action_type == "schedule_maintenance":
            if not has_valid_evidence:
                is_unsafe = True

        if action_type == "verify_sensor":
            if not has_telemetry_or_sensor_evidence(evidence, valid_refs):
                is_unsafe = True

        if is_unsafe:
            unsafe_actions += 1

        if high_risk_violation:
            high_risk_violation_indices.append(idx)

        if action_type in generic_action_types:
            broad_target = is_broad_or_missing_target(
                action.get("target", ""),
                generic_action_rules,
            )

            requires_no_valid_evidence = bool(
                (generic_action_rules or {})
                .get("generic_if", {})
                .get("requires_no_valid_rationale_evidence", True)
            )

            if broad_target and (not has_valid_evidence or requires_no_valid_evidence):
                generic_actions += 1

    return ActionEvaluationResult(
        total_actions=total_actions,
        evidence_linked_actions=evidence_linked_actions,
        unsafe_actions=unsafe_actions,
        generic_actions=generic_actions,
        action_without_evidence_indices=action_without_evidence_indices,
        high_risk_violation_indices=high_risk_violation_indices,
    )
    
