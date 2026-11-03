"""
M4 patched harness.

This module applies a validated M4 candidate patch JSON to the M3-style
verified diagnostic harness.

Purpose:
- Use M3 as the base harness.
- Apply patch-level changes to:
  1. source policy notes
  2. evidence table format constraints
  3. diagnosis/evidence/action prompt instructions
  4. action filter rules
  5. repair prompt instructions
  6. repair trigger thresholds

Important guardrails:
- The patch must not expose hidden proxy 'supports' annotations to the LLM input.
- The patch must not modify datasets, gold labels, evaluator, metric definitions,
  proxy evidence builders, test split, or candidate labels.
- The patch is applied only as harness configuration around a fixed base LLM.

This file does not implement final candidate selection. It only defines how to
execute one validated M4 patch.
"""

from __future__ import annotations
from experiments.llm.observable_records import format_observation_records

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from experiments.evaluation.deterministic_evaluator import (
    evaluate_report,
    evaluation_result_to_dict,
)
from experiments.harness.base import HarnessResult
from experiments.evaluation.runtime_verifier import verify_runtime_report, runtime_failure_summary, require_runtime_result
from experiments.data.generation_evidence import GENERATION_COLUMNS, validate_generation_evidence, validate_observation_windows
from experiments.utils.provenance import runtime_provenance
from experiments.llm.openai_client import OpenAIClient
from experiments.llm.prompts import (
    build_system_prompt,
    build_m3_prompt,
    build_m3_repair_prompt,
)


# ============================================================
# Constants
# ============================================================

DEFAULT_M4_EVIDENCE_COLUMNS = [
    "evidence_id",
    "source",
    "field",
    "statistic",
    "direction",
    "value",
    "raw_value",
    "window",
    "event_time",
    "strength",
    "role",
]

FORBIDDEN_LLM_INPUT_COLUMNS = {
    "supports",
    "gold_label",
    "gold",
    "ground_truth",
    "target_label",
    "true_label",
}

DEFAULT_REPAIR_TRIGGERS = {
    "schema_invalid": True,
    "F5": True,
    "F6": True,
    "F7": True,
    "F8": True,
    "unsafe_recommendation_rate_gt": 0.0,
}


# ============================================================
# Patch I/O and validation
# ============================================================

def load_patch(patch_path: str | Path) -> dict[str, Any]:
    path = Path(patch_path)

    if not path.exists():
        raise FileNotFoundError(f"M4 patch JSON not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        patch = json.load(f)

    if not isinstance(patch, dict):
        raise TypeError(f"M4 patch must be a JSON object: {path}")

    validate_patch_for_m4_execution(patch)

    return patch


def validate_patch_for_m4_execution(patch: dict[str, Any]) -> None:
    """
    Runtime guardrail check.

    This is intentionally redundant with candidate_validator.py. The runner
    should never execute a patch that violates anti-leakage or frozen-module
    constraints.
    """
    from experiments.optimization.patch_contract import patch_contract_violations
    violations = patch_contract_violations(patch)
    if violations:
        raise ValueError('Patch contract violation: ' + '; '.join(v['message'] for v in violations))
    evidence_format = patch.get("evidence_table_format", {})
    proposed_columns = evidence_format.get("proposed_columns", [])
    forbidden_columns = evidence_format.get("forbidden_columns", [])

    proposed_set = {str(c) for c in proposed_columns}
    forbidden_set = {str(c) for c in forbidden_columns}

    leaked_columns = proposed_set & FORBIDDEN_LLM_INPUT_COLUMNS
    if leaked_columns:
        raise ValueError(
            f"M4 patch attempts to expose forbidden input columns: {sorted(leaked_columns)}"
        )

    if "supports" not in forbidden_set:
        raise ValueError(
            "M4 patch must explicitly include 'supports' in evidence_table_format.forbidden_columns."
        )

    frozen_modules = set(patch.get("frozen_modules", []))
    required_frozen = {
        "datasets",
        "gold_labels",
        "proxy_evidence_builder",
        "evaluator",
        "metric_definitions",
        "test_split",
        "candidate_labels",
        "hidden_supports_annotations",
    }
    missing = required_frozen - frozen_modules

    if missing:
        raise ValueError(
            f"M4 patch is missing required frozen modules: {sorted(missing)}"
        )


# ============================================================
# Patch formatting helpers
# ============================================================

def _list_lines(items: list[Any] | None, *, prefix: str = "-") -> str:
    if not items:
        return "- None"

    return "\n".join(f"{prefix} {str(item)}" for item in items)


def _dict_section(data: dict[str, Any] | None) -> str:
    if not data:
        return "None"
    return json.dumps(data, ensure_ascii=False, indent=2, default=str)


def get_m4_evidence_columns(patch: dict[str, Any]) -> list[str]:
    columns = (
        patch
        .get("evidence_table_format", {})
        .get("proposed_columns", DEFAULT_M4_EVIDENCE_COLUMNS)
    )

    safe_columns = []

    for col in columns:
        col = str(col)
        if col in FORBIDDEN_LLM_INPUT_COLUMNS or col not in GENERATION_COLUMNS:
            continue
        if col not in safe_columns:
            safe_columns.append(col)

    if not safe_columns:
        safe_columns = list(DEFAULT_M4_EVIDENCE_COLUMNS)

    return safe_columns


def format_m4_evidence_table(
    proxy_evidence: pd.DataFrame | list[dict[str, Any]] | None,
    patch: dict[str, Any],
    *,
    max_rows: int = 10,
) -> str:
    """
    Format evidence table for LLM input.

    The hidden 'supports' annotation is never exposed.
    """
    columns = get_m4_evidence_columns(patch)

    return format_observation_records(
        proxy_evidence,
        max_rows=max_rows,
        columns=columns,
    )


def format_m4_patch_instructions(patch: dict[str, Any]) -> str:
    source_policy = patch.get("source_policy", {})
    evidence_format = patch.get("evidence_table_format", {})
    prompt_instructions = patch.get("prompt_instructions", {})
    action_filter_rules = patch.get("action_filter_rules", {})

    diagnosis_instructions = prompt_instructions.get("diagnosis_instructions", [])
    evidence_instructions = prompt_instructions.get("evidence_instructions", [])
    action_instructions = prompt_instructions.get("action_instructions", [])

    source_changes = source_policy.get("proposed_changes", [])
    evidence_rules = evidence_format.get("formatting_rules", [])
    action_rules = action_filter_rules.get("rules", [])
    out_of_scope = action_filter_rules.get("out_of_scope_actions", [])

    return f"""
M4 patch instructions:

Source policy:
{_list_lines(source_changes)}

Evidence table rules:
{_list_lines(evidence_rules)}

Diagnosis instructions:
{_list_lines(diagnosis_instructions)}

Evidence instructions:
{_list_lines(evidence_instructions)}

Action instructions:
{_list_lines(action_instructions)}

Action filter rules:
{_list_lines(action_rules)}

Out-of-scope actions:
{_list_lines(out_of_scope)}
""".strip()

def format_m4_optimized_association_policy(
    association_guide: dict[str, Any] | None,
) -> str:
    """
    Legacy M4-specific source-interpretation policy, retained for audit only.

    Historical prompts used this block to make M4 use Azure associations more
    explicitly than M2/M3. Current builders do not inject it: any optimized
    interpretation must now be proposed in the patch, with the common M3 base.
    """
    if not association_guide or association_guide.get("dataset") != "Azure PdM":
        return "No M4-specific Azure association policy is applicable."

    high_conf_error_items = []
    for item in association_guide.get("top_associations", []) or []:
        if item.get("source") != "errors":
            continue
        if item.get("confidence") != "high":
            continue

        evidence_key = item.get("evidence_key", "")
        top_component = item.get("top_component", "")
        prob = item.get("top_probability", "")
        n = item.get("n", "")

        high_conf_error_items.append(
            f"- {evidence_key}: use as a strong weak prior for {top_component} "
            f"when the current evidence table contains the same errorID "
            f"(p={prob}, n={n}, search split only)."
        )

    if not high_conf_error_items:
        high_conf_error_items.append(
            "- No high-confidence errorID association is available; rely on current-task evidence and report low confidence when ambiguous."
        )

    return f"""
M4 optimized Azure association policy:
- Use the search-derived association guide as a source-interpretation prior, not as task-level ground truth.
- For Azure PdM, high-confidence errorID associations may be used as strong weak priors for predicted_label selection when the current evidence table contains the same errorID.
- Do not use validation labels, test labels, task-level gold labels, or hidden supports.
- Do not treat maintenance associations as causal. Maintenance associations remain contextual and may only inform component plausibility.
- If high-confidence errorID evidence conflicts with other current-task evidence, state the ambiguity and use lower confidence.
- Do not invent evidence IDs, source fields, or support annotations.

High-confidence errorID priors:
{chr(10).join(high_conf_error_items)}
""".strip()


def format_m4_repair_instructions(patch: dict[str, Any]) -> str:
    repair_prompt = patch.get("repair_prompt", {})
    repair_instructions = repair_prompt.get("repair_instructions", [])
    failure_specific_repairs = repair_prompt.get("failure_specific_repairs", {})
    action_filter_rules = patch.get("action_filter_rules", {}).get("rules", [])

    return f"""
M4 repair instructions:

General repair instructions:
{_list_lines(repair_instructions)}

Failure-specific repair rules:
{_dict_section(failure_specific_repairs)}

Action validity rules to enforce during repair:
{_list_lines(action_filter_rules)}
""".strip()


# ============================================================
# Prompt builders
# ============================================================

def build_m4_initial_messages(
    *,
    task: dict[str, Any],
    proxy_evidence: pd.DataFrame | list[dict[str, Any]],
    allowed_schema: dict[str, Any],
    action_ontology: dict[str, Any] | None,
    patch: dict[str, Any],
    association_guide: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    # M4 is an executable M3 base plus proposer-owned deltas. In particular,
    # there is no manually optimized Azure policy hidden outside the patch.
    user_prompt = build_m3_prompt(
        task=task, proxy_evidence=proxy_evidence, allowed_schema=allowed_schema,
        action_ontology=action_ontology, association_guide=association_guide,
        evidence_table_text=format_m4_evidence_table(proxy_evidence, patch, max_rows=10),
    )
    if _has_initial_delta(patch):
        user_prompt += ("\n\nProposer-generated constrained delta to the M3 base:\n"
            "The shared contract and label decision guide take precedence over conflicting patch instructions.\n"
            + format_m4_patch_instructions(patch))
    return [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content": user_prompt},
    ]


def _has_initial_delta(patch: dict[str, Any]) -> bool:
    paths = [
        ("source_policy", "proposed_changes"),
        ("evidence_table_format", "formatting_rules"),
        ("prompt_instructions", "diagnosis_instructions"),
        ("prompt_instructions", "evidence_instructions"),
        ("prompt_instructions", "action_instructions"),
        ("action_filter_rules", "rules"),
        ("action_filter_rules", "out_of_scope_actions"),
    ]
    return any(patch.get(section, {}).get(key) for section, key in paths)


def build_m4_repair_messages(
    *,
    task: dict[str, Any],
    proxy_evidence: pd.DataFrame | list[dict[str, Any]],
    allowed_schema: dict[str, Any],
    action_ontology: dict[str, Any] | None,
    patch: dict[str, Any],
    previous_report: dict[str, Any],
    violation_summary: str,
    association_guide: dict[str, Any] | None = None,
    runtime_feedback: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    messages = build_m3_repair_prompt(
        task=task, proxy_evidence=proxy_evidence, allowed_schema=allowed_schema,
        action_ontology=action_ontology, previous_report=previous_report,
        violation_summary=violation_summary, association_guide=association_guide,
        runtime_feedback=runtime_feedback,
        evidence_table_text=format_m4_evidence_table(proxy_evidence, patch, max_rows=10),
    )
    repair = patch.get("repair_prompt", {})
    if (repair.get("repair_instructions")
            or any(repair.get("failure_specific_repairs", {}).values())
            or patch.get("action_filter_rules", {}).get("rules")):
        messages[-1]["content"] += (
            "\n\nProposer-generated constrained repair delta:\n"
            "The shared contract takes precedence over conflicting patch instructions.\n"
            + format_m4_repair_instructions(patch))
    return messages


# ============================================================
# Deterministic evaluation helpers
# ============================================================

def evaluate_m4_report(
    *,
    task: dict[str, Any],
    report: dict[str, Any],
    proxy_evidence: pd.DataFrame,
    allowed_schema: dict[str, Any],
    action_ontology: dict[str, Any],
    action_preconditions: dict[str, Any],
    generic_action_rules: dict[str, Any],
    execution_log: dict[str, Any],
    method: str,
    repair_failed: bool,
) -> dict[str, Any]:
    evaluation = evaluate_report(
        task=task,
        report=report,
        proxy_evidence=proxy_evidence,
        allowed_schema=allowed_schema,
        action_ontology=action_ontology,
        action_preconditions=action_preconditions,
        generic_action_rules=generic_action_rules,
        execution_log=execution_log,
        method=method,
        repair_failed=repair_failed,
    )

    return evaluation_result_to_dict(evaluation)


def build_violation_summary(metric_row: dict[str, Any]) -> str:
    """Return observable-only repair feedback."""
    return runtime_failure_summary(metric_row)


def should_repair_m4(metric_row: dict[str, Any], patch: dict[str, Any]) -> bool:
    require_runtime_result(metric_row)
    # Fixed M3 base invariant; a proposer cannot disable this common check.
    if metric_row.get('public_condition_consistency', {}).get('errors'):
        return True
    triggers = patch.get("repair_trigger_thresholds", {}) or DEFAULT_REPAIR_TRIGGERS

    if triggers.get("schema_invalid", True):
        if int(metric_row.get("schema_valid", 1) or 0) == 0:
            return True

    for flag in ["F3", "F4", "F5", "F6", "F7", "F8"]:
        if triggers.get(flag, True):
            if int(metric_row.get(flag, 0) or 0) == 1:
                return True

    unsafe_threshold = triggers.get("unsafe_recommendation_rate_gt", None)
    if unsafe_threshold is not None:
        unsafe = float(metric_row.get("unsafe_recommendation_rate", 0.0) or 0.0)
        if unsafe > float(unsafe_threshold):
            return True

    unsupported_threshold = triggers.get("unsupported_claim_rate_gt", None)
    if unsupported_threshold is not None:
        unsupported = float(metric_row.get("unsupported_claim_rate", 0.0) or 0.0)
        if unsupported > float(unsupported_threshold):
            return True

    return False


# ============================================================
# Execution log helpers
# ============================================================

def infer_accessed_sources_for_m4(proxy_evidence: pd.DataFrame) -> list[str]:
    if proxy_evidence is None or proxy_evidence.empty or "source" not in proxy_evidence.columns:
        return []

    return sorted(set(proxy_evidence["source"].dropna().astype(str).tolist()))


def build_m4_tool_calls(*, patch: dict[str, Any], repair_count: int) -> list[dict[str, Any]]:
    calls = [
        {
            "tool_name": "m4_patch_loader",
            "status": "ok",
            "candidate_id": patch.get("candidate_id"),
        },
        {
            "tool_name": "source_policy_lookup",
            "status": "ok",
        },
        {
            "tool_name": "diagnostic_evidence_table",
            "status": "ok",
            "columns": get_m4_evidence_columns(patch),
        },
        {
            "tool_name": "deterministic_verifier",
            "status": "ok",
        },
        {
            "tool_name": "action_validity_filter",
            "status": "ok",
        },
    ]

    if repair_count > 0:
        calls.append(
            {
                "tool_name": "m4_one_step_repair",
                "status": "ok",
            }
        )

    return calls


def make_execution_log_for_evaluation(
    *,
    method: str,
    harness_id: str,
    accessed_sources: list[str],
    tool_calls: list[dict[str, Any]],
    repair_count: int,
    latency_sec: float,
    input_tokens: int,
    output_tokens: int,
    cost_usd: float,
) -> dict[str, Any]:
    return {
        "method": method,
        "harness_id": harness_id,
        "accessed_sources": accessed_sources,
        "tool_calls": tool_calls,
        "repair_count": repair_count,
        "latency_sec": latency_sec,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": cost_usd,
    }


# ============================================================
# M4 harness
# ============================================================

@dataclass
class M4PatchedHarness:
    patch: dict[str, Any]
    client: OpenAIClient | None = None
    dummy_mode: bool = False

    @classmethod
    def from_patch_path(
        cls,
        patch_path: str | Path,
        *,
        client: OpenAIClient | None = None,
        dummy_mode: bool = False,
    ) -> "M4PatchedHarness":
        patch = load_patch(patch_path)
        return cls(
            patch=patch,
            client=client,
            dummy_mode=dummy_mode,
        )

    @property
    def method(self) -> str:
        return "M4"

    @property
    def harness_id(self) -> str:
        return str(self.patch.get("candidate_id", "M4_patched_candidate"))

    def run(
        self,
        *,
        task: dict[str, Any],
        proxy_evidence: pd.DataFrame,
        allowed_schema: dict[str, Any],
        action_ontology: dict[str, Any],
        action_preconditions: dict[str, Any],
        generic_action_rules: dict[str, Any],
        association_guide: dict[str, Any] | None = None,
    ) -> HarnessResult:
        if self.dummy_mode:
            return self._run_dummy(
                task=task,
                proxy_evidence=proxy_evidence,
            )

        if self.client is None:
            raise RuntimeError("M4PatchedHarness requires OpenAIClient unless dummy_mode=True.")

        return self._run_api(
            task=task,
            proxy_evidence=proxy_evidence,
            allowed_schema=allowed_schema,
            action_ontology=action_ontology,
            action_preconditions=action_preconditions,
            generic_action_rules=generic_action_rules,
            association_guide=association_guide,
        )

    def _run_dummy(
        self,
        *,
        task: dict[str, Any],
        proxy_evidence: pd.DataFrame,
    ) -> HarnessResult:
        """
        Minimal dry-run behavior for integration tests only.
        """
        candidate_labels = []
        if isinstance(task, dict):
            if "candidate_labels" in task and isinstance(task["candidate_labels"], list):
                candidate_labels = task["candidate_labels"]

        predicted_label = candidate_labels[0] if candidate_labels else "comp1" if "azure" in str(task.get("dataset", "")).lower() else "No Failure"

        report = {
            "predicted_label": predicted_label,
            "confidence": "low",
            "evidence": [],
            "next_steps": [
                {
                    "action_type": "monitor",
                    "target": "machine",
                    "rationale_evidence_ids": [],
                }
            ],
            "limitations": [
                "M4 dummy-mode report. No API call was made."
            ],
        }

        return HarnessResult(
            method="M4",
            harness_id=self.harness_id,
            task_id=str(task.get("task_id", "")),
            dataset=str(task.get("dataset", "")),
            report=report,
            accessed_sources=[],
            tool_calls=[
                {"tool_name": "m4_dummy_run", "status": "ok"},
            ],
            repair_count=0,
            repair_failed=False,
            latency_sec=0.0,
            input_tokens=0,
            output_tokens=0,
            cost_usd=0.0,
            model_name="dummy",
            raw_response={
                "dummy_mode": True,
                "patch_candidate_id": self.harness_id,
            },
        )

    def _run_api(
        self,
        *,
        task: dict[str, Any],
        proxy_evidence: pd.DataFrame,
        allowed_schema: dict[str, Any],
        action_ontology: dict[str, Any],
        action_preconditions: dict[str, Any],
        generic_action_rules: dict[str, Any],
        association_guide: dict[str, Any] | None = None,
    ) -> HarnessResult:
        validate_patch_for_m4_execution(self.patch)
        validate_generation_evidence(proxy_evidence, [task["task_id"]])
        validate_observation_windows(proxy_evidence, pd.DataFrame([task]))

        start_total = time.perf_counter()

        accessed_sources = infer_accessed_sources_for_m4(proxy_evidence)

        # First generation
        first_messages = build_m4_initial_messages(
            task=task,
            proxy_evidence=proxy_evidence,
            allowed_schema=allowed_schema,
            action_ontology=action_ontology,
            patch=self.patch,
            association_guide=association_guide,
        )

        first_result = self.client.generate_structured_report(messages=first_messages)

        repair_count = 0
        repair_failed = False
        final_report = first_result.parsed_report
        final_raw_response: Any = first_result.raw_response
        final_input_tokens = first_result.input_tokens
        final_output_tokens = first_result.output_tokens
        final_cost = first_result.cost_usd
        final_latency = first_result.latency_sec

        provisional_tool_calls = build_m4_tool_calls(
            patch=self.patch,
            repair_count=0,
        )

        provisional_execution_log = make_execution_log_for_evaluation(
            method="M4",
            harness_id=self.harness_id,
            accessed_sources=accessed_sources,
            tool_calls=provisional_tool_calls,
            repair_count=0,
            latency_sec=first_result.latency_sec,
            input_tokens=first_result.input_tokens,
            output_tokens=first_result.output_tokens,
            cost_usd=first_result.cost_usd,
        )

        provisional_metric_row = verify_runtime_report(
            report=first_result.parsed_report,
            generation_evidence=proxy_evidence,
            allowed_schema=allowed_schema,
            action_ontology=action_ontology,
            action_preconditions=action_preconditions,
            generic_action_rules=generic_action_rules,
        )

        # One-step repair if patch trigger fires.
        if should_repair_m4(provisional_metric_row, self.patch):
            violation_summary = build_violation_summary(provisional_metric_row)

            repair_messages = build_m4_repair_messages(
                task=task,
                proxy_evidence=proxy_evidence,
                allowed_schema=allowed_schema,
                action_ontology=action_ontology,
                patch=self.patch,
                previous_report=first_result.parsed_report,
                violation_summary=violation_summary,
                runtime_feedback=provisional_metric_row,
                association_guide=association_guide,
            )

            repair_result = self.client.generate_structured_report(messages=repair_messages)

            repair_count = 1
            final_report = repair_result.parsed_report

            final_input_tokens = first_result.input_tokens + repair_result.input_tokens
            final_output_tokens = first_result.output_tokens + repair_result.output_tokens
            final_cost = first_result.cost_usd + repair_result.cost_usd
            final_latency = first_result.latency_sec + repair_result.latency_sec

            repaired_tool_calls = build_m4_tool_calls(
                patch=self.patch,
                repair_count=repair_count,
            )

            repaired_execution_log = make_execution_log_for_evaluation(
                method="M4",
                harness_id=self.harness_id,
                accessed_sources=accessed_sources,
                tool_calls=repaired_tool_calls,
                repair_count=repair_count,
                latency_sec=final_latency,
                input_tokens=final_input_tokens,
                output_tokens=final_output_tokens,
                cost_usd=final_cost,
            )

            repaired_metric_row = verify_runtime_report(
                report=final_report,
                generation_evidence=proxy_evidence,
                allowed_schema=allowed_schema,
                action_ontology=action_ontology,
                action_preconditions=action_preconditions,
                generic_action_rules=generic_action_rules,
            )

            repair_failed = should_repair_m4(repaired_metric_row, self.patch)

            final_raw_response = {
                "first_response": first_result.raw_response,
                "repair_response": repair_result.raw_response,
                "violation_summary": violation_summary,
                "provisional_metric_row": provisional_metric_row,
                "repaired_metric_row": repaired_metric_row,
            }
        from experiments.evaluation.runtime_verifier import audit_runtime_repair
        from experiments.harness.verified_postprocessing import finalize_verified_report
        pre_filter_report = final_report
        postprocessing = finalize_verified_report(
            report=pre_filter_report, generation_evidence=proxy_evidence,
            allowed_schema=allowed_schema, action_ontology=action_ontology,
            action_preconditions=action_preconditions, generic_action_rules=generic_action_rules,
        )
        repair_attempt_failed = should_repair_m4(postprocessing['pre_filter_verification'], self.patch) if repair_count else None
        final_report = postprocessing['report']
        repair_failed = bool(repair_count and should_repair_m4(postprocessing['final_runtime_verification'], self.patch))
        final_raw_response = {
            "responses": final_raw_response, "initial_messages": first_messages,
            "initial_report": first_result.parsed_report,
            "repair_report": pre_filter_report if repair_count else None,
            "pre_filter_report": pre_filter_report,
            "action_filter": postprocessing['action_filter'],
            "pre_filter_runtime_verification": postprocessing['pre_filter_verification'],
            "repair_attempt_failed": repair_attempt_failed,
            "repair_messages": repair_messages if repair_count else None,
            "initial_runtime_verification": provisional_metric_row,
            "final_runtime_verification": postprocessing['final_runtime_verification'],
            "repair_metadata_audit": audit_runtime_repair(first_result.parsed_report,pre_filter_report,proxy_evidence) if repair_count else None,
        }

        elapsed_total = time.perf_counter() - start_total

        final_tool_calls = build_m4_tool_calls(
            patch=self.patch,
            repair_count=repair_count,
        )

        return HarnessResult(
            method="M4",
            harness_id=self.harness_id,
            task_id=str(task.get("task_id", "")),
            dataset=str(task.get("dataset", "")),
            report=final_report,
            accessed_sources=accessed_sources,
            tool_calls=final_tool_calls,
            repair_count=repair_count,
            repair_failed=repair_failed,
            latency_sec=max(final_latency, elapsed_total),
            input_tokens=final_input_tokens,
            output_tokens=final_output_tokens,
            cost_usd=final_cost,
            model_name=self.client.model,
            raw_response=final_raw_response,
            extra={"protocol": runtime_provenance(), "action_filter": postprocessing['action_filter'],
                   "repair_attempt_failed": repair_attempt_failed},
        )


# ============================================================
# Convenience function
# ============================================================

def run_m4_patched_harness(
    *,
    patch_path: str | Path,
    client: OpenAIClient,
    task: dict[str, Any],
    proxy_evidence: pd.DataFrame,
    allowed_schema: dict[str, Any],
    action_ontology: dict[str, Any],
    action_preconditions: dict[str, Any],
    generic_action_rules: dict[str, Any],
    association_guide: dict[str, Any] | None = None,
) -> HarnessResult:
    harness = M4PatchedHarness.from_patch_path(
        patch_path,
        client=client,
        dummy_mode=False,
    )

    return harness.run(
        task=task,
        proxy_evidence=proxy_evidence,
        allowed_schema=allowed_schema,
        action_ontology=action_ontology,
        action_preconditions=action_preconditions,
        generic_action_rules=generic_action_rules,
        association_guide=association_guide,
    )
