"""
End-to-end deterministic evaluator.

This module evaluates one structured diagnostic report for one diagnostic task.
It produces task-level metrics and F1--F10 failure flags. Dataset-level metrics
such as Macro-F1 are aggregated later from task-level predictions.

This is offline scoring with reference labels/targets. Runtime generation and
repair must instead use runtime_verifier.verify_runtime_report.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .action_evaluator import evaluate_actions
from .evidence_matcher import match_generated_evidence
from .failure_trace_tagger import tag_failure_traces
from .structured_report_schema import normalize_label, validate_report_schema
from .metric_policy import EVALUATION_POLICY_VERSION


@dataclass(frozen=True)
class EvaluationResult:
    task_id: str
    dataset: str
    method: str

    gold_label: str
    predicted_label: str
    label_correct: int

    evidence_precision: float | None
    evidence_recall: float | None

    unsupported_claim_rate: float
    unsafe_recommendation_rate: float
    generic_action_rate: float
    cost_per_report: float

    source_hallucination_rate: float
    temporal_violation_rate: float
    evidence_linked_action_ratio: float | None
    causal_overclaim_rate: float

    schema_valid: int
    schema_errors: list[str]

    failure_flags: dict[str, int]

    n_generated_evidence: int
    n_required_proxy_evidence: int
    n_generated_actions: int
    n_unsupported_claims: int
    evidence_present: int
    required_evidence_present: int
    action_present: int
    evidence_linked_action_present: int
    n_evidence_linked_actions: int
    n_unsafe_actions: int
    evaluation_policy_version: str
    report_contract_version: str | None = None


def estimate_cost_from_execution_log(
    execution_log: dict[str, Any] | None,
) -> float:
    """
    Read cost from execution log.

    Cost calculation should be performed by the API client or runner because
    model prices may change. This evaluator only reads the logged value.
    """
    if not execution_log:
        return 0.0

    if execution_log.get("cost_usd") is not None:
        return float(execution_log["cost_usd"])

    if execution_log.get("api_cost") is not None:
        return float(execution_log["api_cost"])

    return 0.0


def detect_excessive_source_access(
    execution_log: dict[str, Any] | None,
    matched_evidence_count: int,
) -> bool:
    """
    Deterministic F9 rule.

    F9 is triggered if:
    - source_access_count exceeds allowed_source_access_count, or
    - sources were accessed but no evidence matched.

    This rule can be tightened later but should remain frozen during a run.
    """
    if not execution_log:
        return False

    source_access_count = int(execution_log.get("source_access_count", 0) or 0)
    allowed_source_access_count = execution_log.get("allowed_source_access_count")

    if allowed_source_access_count is not None:
        if source_access_count > int(allowed_source_access_count):
            return True

    if source_access_count > 0 and matched_evidence_count == 0:
        return True

    return False


def compute_unsupported_claim_rate(
    *,
    n_claims: int,
    invalid_evidence_ids: set[str],
    invalid_action_indices: list[int],
    n_invalid_evidence: int | None = None,
) -> tuple[float, int]:
    """
    Unsupported claims include invalid evidence claims and invalid action rationales.
    Denominator = generated evidence claims + generated action suggestions.
    """
    unsupported_count = (len(invalid_evidence_ids) if n_invalid_evidence is None else n_invalid_evidence) + len(set(invalid_action_indices))

    if n_claims == 0:
        return 0.0, unsupported_count

    return unsupported_count / n_claims, unsupported_count


def evaluate_report(
    *,
    task: dict[str, Any],
    report: dict[str, Any],
    proxy_evidence: Any,
    allowed_schema: dict[str, Any],
    action_ontology: dict[str, Any],
    action_preconditions: dict[str, Any] | None = None,
    generic_action_rules: dict[str, Any] | None = None,
    alias_map: dict[str, list[str]] | None = None,
    execution_log: dict[str, Any] | None = None,
    method: str = "",
    repair_failed: bool = False,
) -> EvaluationResult:
    """
    Evaluate one report deterministically.

    Parameters
    ----------
    task:
        Dict with at least task_id, dataset, gold_label.
    report:
        Structured diagnostic report.
    proxy_evidence:
        Proxy evidence rows for the task. Can be list[dict] or pandas DataFrame.
    allowed_schema:
        Dataset-specific allowed source/field schema.
    action_ontology:
        Allowed action types and risk levels.
    action_preconditions:
        Action precondition configuration. Kept for interface stability.
    generic_action_rules:
        Deterministic generic-action definition.
    alias_map:
        Optional field alias map.
    execution_log:
        Token/cost/source-access/repair log.
    method:
        Method name, e.g., M0, M1, M2, M3, M4.
    repair_failed:
        Whether the report remained invalid after repair.
    """
    candidate_labels = allowed_schema.get("candidate_labels", [])
    allowed_sources = allowed_schema.get("allowed_sources", {})
    allowed_action_types = list(action_ontology.get("action_types", {}).keys())

    schema_result = validate_report_schema(
        report=report,
        candidate_labels=candidate_labels,
        allowed_sources=allowed_sources,
        allowed_action_types=allowed_action_types,
        alias_map=alias_map,
    )

    evidence_items = report.get("evidence", []) if isinstance(report, dict) else []
    if not isinstance(evidence_items, list):
        evidence_items = []

    evidence_match_result = match_generated_evidence(
        generated_evidence=evidence_items,
        proxy_evidence=proxy_evidence,
        allowed_schema=allowed_schema,
        alias_map=alias_map,
        include_context_for_precision=True,
        include_context_in_recall=False,
    )

    action_eval_result = evaluate_actions(
        report=report,
        matched_generated_evidence_ids=evidence_match_result.matched_generated_ids,
        action_ontology=action_ontology,
        action_preconditions=action_preconditions or {},
        generic_action_rules=generic_action_rules or {},
    )

    predicted_label = normalize_label(report.get("predicted_label", ""))
    gold_label = normalize_label(task.get("gold_label", ""))

    invalid_evidence_ids: set[str] = set()
    invalid_evidence_ids.update(evidence_match_result.invalid_source_or_field_ids)
    invalid_evidence_ids.update(evidence_match_result.value_contradiction_ids)
    invalid_evidence_ids.update(evidence_match_result.temporal_violation_ids)
    invalid_evidence_ids.update(evidence_match_result.causal_overclaim_ids)
    invalid_evidence_ids.update(evidence_match_result.unmatched_generated_ids)

    n_claims = len(evidence_items) + action_eval_result.total_actions

    unsupported_claim_rate, n_unsupported_claims = compute_unsupported_claim_rate(
        n_claims=n_claims,
        invalid_evidence_ids=invalid_evidence_ids,
        invalid_action_indices=action_eval_result.action_without_evidence_indices,
        n_invalid_evidence=sum(str(e.get("evidence_id",f"GEN_{i+1}")) in invalid_evidence_ids for i,e in enumerate(evidence_items)),
    )

    source_hallucination_rate = (
        0.0
        if len(evidence_items) == 0
        else len(evidence_match_result.invalid_source_or_field_ids) / len(evidence_items)
    )

    temporal_violation_rate = (
        0.0
        if len(evidence_items) == 0
        else len(evidence_match_result.temporal_violation_ids) / len(evidence_items)
    )

    causal_overclaim_rate = (
        0.0
        if len(evidence_items) == 0
        else len(evidence_match_result.causal_overclaim_ids) / len(evidence_items)
    )

    excessive_source_access = detect_excessive_source_access(
        execution_log=execution_log,
        matched_evidence_count=len(evidence_match_result.matched_generated_ids),
    )

    failure_flags = tag_failure_traces(
        gold_label=gold_label,
        predicted_label=predicted_label,
        evidence_match_result=evidence_match_result,
        action_eval_result=action_eval_result,
        schema_valid=schema_result.valid,
        repair_failed=repair_failed,
        excessive_source_access=excessive_source_access,
    )

    return EvaluationResult(
        task_id=str(task.get("task_id", "")),
        dataset=str(task.get("dataset", "")),
        method=method,

        gold_label=gold_label,
        predicted_label=predicted_label,
        label_correct=int(gold_label == predicted_label),

        evidence_precision=evidence_match_result.precision,
        evidence_recall=evidence_match_result.recall,

        unsupported_claim_rate=unsupported_claim_rate,
        unsafe_recommendation_rate=action_eval_result.unsafe_recommendation_rate,
        generic_action_rate=action_eval_result.generic_action_rate,
        cost_per_report=estimate_cost_from_execution_log(execution_log),

        source_hallucination_rate=source_hallucination_rate,
        temporal_violation_rate=temporal_violation_rate,
        evidence_linked_action_ratio=action_eval_result.evidence_linked_action_ratio,
        causal_overclaim_rate=causal_overclaim_rate,

        schema_valid=int(schema_result.valid),
        schema_errors=schema_result.errors,

        failure_flags=failure_flags,

        n_generated_evidence=len(evidence_items),
        n_required_proxy_evidence=evidence_match_result.proxy_required_count,
        n_generated_actions=action_eval_result.total_actions,
        n_unsupported_claims=n_unsupported_claims,
        evidence_present=int(bool(evidence_items)),
        required_evidence_present=int(evidence_match_result.proxy_required_count>0),
        action_present=int(action_eval_result.total_actions>0),
        evidence_linked_action_present=int(action_eval_result.evidence_linked_actions>0),
        n_evidence_linked_actions=action_eval_result.evidence_linked_actions,
        n_unsafe_actions=action_eval_result.unsafe_actions,
        evaluation_policy_version=EVALUATION_POLICY_VERSION,
        report_contract_version=((execution_log or {}).get('protocol') or {}).get('report_contract_version'),
    )


def evaluation_result_to_dict(result: EvaluationResult) -> dict[str, Any]:
    """
    Flatten EvaluationResult into a CSV/JSONL-friendly dictionary.
    """
    row = asdict(result)
    flags = row.pop("failure_flags")

    for flag_name, value in flags.items():
        row[flag_name] = value

    return row
