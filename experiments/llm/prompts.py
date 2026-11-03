"""
Prompt builders for M0--M3 diagnostic harnesses.

These functions produce compact, method-specific prompts. They do not call the
LLM. The OpenAI client receives the resulting system/user messages.
"""

from __future__ import annotations
from experiments.llm.report_contract import format_common_contract
from experiments.llm.observable_records import format_observation_records, format_runtime_corrections, format_minimal_repair_contract

from typing import Any
import json

import pandas as pd
from experiments.data.generation_evidence import generation_task, GENERATION_COLUMNS


# ============================================================
# Formatting helpers
# ============================================================

def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    if pd.isna(value):
        return ""
    return str(value)


def dataframe_to_compact_records(
    df: pd.DataFrame | list[dict[str, Any]] | None,
    *,
    max_rows: int = 8,
    columns: list[str] | None = None,
) -> str:
    if df is None:
        return "None"

    if isinstance(df, list):
        df = pd.DataFrame(df)

    if df.empty:
        return "None"

    use_df = df.copy()

    if columns is not None:
        existing_cols = [col for col in columns if col in use_df.columns]
        use_df = use_df[existing_cols]

    use_df = use_df.head(max_rows)
    return use_df.to_string(index=False)


def format_allowed_schema(allowed_schema: dict[str, Any] | None) -> str:
    if not allowed_schema:
        return "No allowed schema provided."

    sources = allowed_schema.get("allowed_sources", {})
    labels = allowed_schema.get("candidate_labels", [])
    roles = allowed_schema.get("evidence_roles", [])
    windows = allowed_schema.get("window_values", [])

    lines = []
    lines.append(f"Candidate labels: {labels}")
    lines.append("Allowed sources and fields:")

    for source, fields in sources.items():
        lines.append(f"- {source}: {fields}")

    if roles:
        lines.append(f"Allowed evidence roles: {roles}")

    if windows:
        lines.append(f"Allowed windows: {windows}")

    return "\n".join(lines)


def format_action_ontology(action_ontology: dict[str, Any] | None) -> str:
    if not action_ontology:
        return "No action ontology provided."

    action_types = action_ontology.get("action_types", {})

    lines = ["Allowed action types:"]
    for action_type, info in action_types.items():
        risk = info.get("risk", "unknown")
        allowed = info.get("allowed", False)
        lines.append(f"- {action_type}: risk={risk}, allowed={allowed}")

    return "\n".join(lines)


def format_task_context(task: dict[str, Any], *, include_gold_label: bool = False, include_auxiliary: bool = False) -> str:
    """
    Format task fields.

    gold_label must not be exposed to the LLM during real experiments.
    It is included only if include_gold_label=True for debugging.
    """
    if include_gold_label:
        raise ValueError("Diagnostic prompts cannot expose gold labels")
    task = generation_task(task)
    lines = []

    for key, value in task.items():
        if key == 'azure_observation_context':
            continue
        if key == "gold_label" and not include_gold_label:
            continue
        lines.append(f"{key}: {_safe_str(value)}")

    if include_auxiliary:
        from experiments.data.azure_observation_context import CONTEXT_KEY, format_context_contract
        from experiments.data.twf_risk import assess_twf_risk
        if CONTEXT_KEY in task:
            lines.extend([format_context_contract(), json.dumps(task[CONTEXT_KEY], separators=(',', ':'), allow_nan=False)])
        if 'tool_wear' in task:
            lines.extend(['Advisory public TWF risk (not a label or calibrated probability):',
                          json.dumps(assess_twf_risk(task['tool_wear']), allow_nan=False)])

    return "\n".join(lines)


def format_proxy_evidence_table(
    proxy_evidence: pd.DataFrame | list[dict[str, Any]] | None,
    *,
    max_rows: int = 8,
) -> str:
    columns = [
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

    return format_observation_records(
        proxy_evidence,
        max_rows=max_rows,
        columns=columns,
    )


def format_label_decision_guide(allowed_schema: dict[str, Any] | None) -> str:
    dataset = (allowed_schema or {}).get("dataset", "")

    if dataset == "AI4I 2020":
        from experiments.data.twf_risk import format_twf_guide
        return """
Dataset label decision guide:
- No Failure: consider when the observable failure-mode indicators do not support a fault. Failure flags are not available.
- TWF: elevated tool wear is a risk indicator, not a deterministic proof of tool-wear failure.
- HDF: heat dissipation failure; the temperature gap is process minus air temperature. Consider HDF when the gap is below 8.6 K and rotational speed is below 1380 rpm.
- PWF: power failure; choose when mechanical_power_w is outside the normal operating range, especially below 3500 W or above 9000 W.
- OSF: compare torque * tool_wear with the variant threshold: L=11000, M=12000, H=13000 minNm.
- Derived feature meanings: temperature_gap = process_temperature - air_temperature; mechanical_power_w = torque * 2*pi*rotational_speed/60; overstrain_score = torque * tool_wear (the product itself, not a normalized score or threshold exceedance).
- A supplied overstrain_score is already the computed product. Compare it directly with the type-specific threshold. A power value within [3500, 9000] W does not satisfy PWF, and overstrain_score at or below its threshold does not satisfy OSF.
- If multiple indicators are present, select the label with the most direct label-specific evidence.
""".strip() + '\n\n' + format_twf_guide()

    if dataset == "Azure PdM":
        return """
Dataset label decision guide:
- Candidate labels comp1, comp2, comp3, and comp4 denote the failed component for the event.
- Telemetry anomalies and error events in the pre-failure window are symptomatic evidence.
- Maintenance records are contextual by default, not causal by themselves.
- If a maintenance evidence value names a component such as comp1, comp2, comp3, or comp4, use it as contextual support for that component but do not overstate it as causal.
- If evidence is ambiguous across components, select the most plausible component, use low confidence, and state the ambiguity.
""".strip()

    return "No dataset-specific label decision guide provided."

'''
def format_azure_association_guide(
    association_guide: dict[str, Any] | None,
    *,
    max_items: int = 12,
) -> str:
    """
    Format search-split-only Azure component association guide.

    This guide is an aggregate weak prior. It must not be interpreted as
    task-level ground truth or hidden support labels.
    """
    if not association_guide:
        return "No search-derived Azure association guide provided."

    if association_guide.get("dataset") != "Azure PdM":
        return "No search-derived Azure association guide provided."

    top = association_guide.get("top_associations", []) or []
    if not top:
        return "No search-derived Azure association guide provided."

    lines = [
        "Search-derived Azure component association guide:",
        "- These associations were computed only from the Azure PdM search split.",
        "- Treat them as weak aggregate priors, not as ground truth for the current task.",
        "- Use an association only when the current evidence table contains the same source/field/value or source/field/direction pattern.",
        "- Do not expose or infer hidden task-level supports.",
        "- Maintenance evidence is contextual by default and should not be described as causal by itself.",
        "Associations:",
    ]

    for item in top[:max_items]:
        evidence_key = item.get("evidence_key", "")
        top_component = item.get("top_component", "")
        n = item.get("n", "")
        prob = item.get("top_probability", "")
        confidence = item.get("confidence", "")
        dist = item.get("component_distribution", {})

        lines.append(
            f"- {evidence_key}: most associated with {top_component} "
            f"in the search split (p={prob}, n={n}, confidence={confidence}, distribution={dist})"
        )

    return "\n".join(lines)
'''

def format_azure_association_guide(
    association_guide: dict[str, Any] | None,
    *,
    max_items: int = 12,
) -> str:
    """
    Format search-split-only Azure component association guide for M0--M3 prompts.

    For M2/M3, this guide should be treated as weak background context,
    not as a deterministic label-selection rule.
    """
    if not association_guide:
        return "No search-derived Azure association guide provided."

    if association_guide.get("dataset") != "Azure PdM":
        return "No search-derived Azure association guide provided."

    top = association_guide.get("top_associations", []) or []
    if not top:
        return "No search-derived Azure association guide provided."

    lines = [
        "Search-derived Azure component association guide:",
        "- These associations were computed only from the Azure PdM search split.",
        "- Treat them as weak background priors, not as deterministic label rules.",
        "- Do not select a component solely because it has the highest association probability.",
        "- Use an association only when the current evidence table contains the same source/field/value or source/field/direction pattern.",
        "- Prefer current-task evidence over aggregate associations.",
        "- If current-task evidence is ambiguous or conflicts with the guide, select the most plausible component with low confidence.",
        "- Do not expose or infer hidden task-level supports.",
        "- Maintenance evidence is contextual by default and should not be described as causal by itself.",
        "Associations:",
    ]

    for item in top[:max_items]:
        evidence_key = item.get("evidence_key", "")
        top_component = item.get("top_component", "")
        n = item.get("n", "")
        prob = item.get("top_probability", "")
        confidence = item.get("confidence", "")
        dist = item.get("component_distribution", {})

        lines.append(
            f"- {evidence_key}: most associated with {top_component} "
            f"in the search split (p={prob}, n={n}, confidence={confidence}, distribution={dist})"
        )

    return "\n".join(lines)

# ============================================================
# System prompt
# ============================================================

def build_system_prompt() -> str:
    return (
        "You are a manufacturing fault-diagnosis assistant. "
        "Generate a structured diagnostic report for human engineering review. "
        "Do not claim evidence that is not present in the provided context. "
        "Do not recommend autonomous control, parameter adjustment, or shutdown. "
        "Every action must be linked to valid evidence IDs when evidence is available. "
        "Return only the structured report that conforms to the required schema."
    )


# ============================================================
# Method-specific user prompts
# ============================================================

def build_m0_prompt(
    *,
    task: dict[str, Any],
    allowed_schema: dict[str, Any],
    action_ontology: dict[str, Any] | None = None,
) -> str:
    """
    M0: minimal structured LLM baseline.
    No proxy evidence table is provided.
    """
    return f"""
Method: M0 Zero-shot structured diagnostic report.

Task:
{format_task_context(task)}

Allowed diagnostic schema:
{format_allowed_schema(allowed_schema)}

Action ontology:
{format_action_ontology(action_ontology)}

Instructions:
- Select predicted_label from the candidate labels only.
- Use evidence only if it is directly available in the task fields.
- If evidence is insufficient, keep evidence minimal and state limitations.
- Avoid high-risk or out-of-scope actions.
- Return a structured diagnostic report.
""".strip()


def build_m1_prompt(
    *,
    task: dict[str, Any],
    allowed_schema: dict[str, Any],
    action_ontology: dict[str, Any] | None = None,
) -> str:
    """
    M1: static diagnostic summary.
    Uses task fields and derived/static features.
    """
    return f"""
Method: M1 Static diagnostic summary harness.

Task and static diagnostic summary:
{format_task_context(task)}

Allowed diagnostic schema:
{format_allowed_schema(allowed_schema)}

Label decision guide:
{format_label_decision_guide(allowed_schema)}

Action ontology:
{format_action_ontology(action_ontology)}

Instructions:
- Use the task fields and derived/static indicators as the diagnostic context.
- Select predicted_label from the candidate labels only.
- Cite only sources and fields listed in the allowed schema.
- Evidence should use available task fields or derived indicators.
- Recommended actions must be conservative and evidence-linked when possible.
- Return a structured diagnostic report.
""".strip()


def build_source_evidence_prompt(
    *,
    task: dict[str, Any],
    proxy_evidence: pd.DataFrame | list[dict[str, Any]],
    allowed_schema: dict[str, Any],
    action_ontology: dict[str, Any] | None = None,
    association_guide: dict[str, Any] | None = None,
    evidence_table_text: str | None = None,
) -> str:
    """Identical competent first-generation context for M2 and M3.

    Method identity remains in execution logs, not in model-facing messages.
    All source/diagnosis/format/action guidance is shared. Only M3 adds actual
    post-generation verification, one repair, and deterministic action filtering.
    """
    return f"""
Task: Source-aware structured manufacturing diagnostic report.

{format_common_contract()}

Task:
{format_task_context(task, include_auxiliary=True)}

Allowed diagnostic schema:
{format_allowed_schema(allowed_schema)}

Label decision guide:
{format_label_decision_guide(allowed_schema)}

Azure search-derived association guide:
{format_azure_association_guide(association_guide)}

Diagnostic evidence table:
{evidence_table_text if evidence_table_text is not None else format_proxy_evidence_table(proxy_evidence, max_rows=10)}

Action ontology:
{format_action_ontology(action_ontology)}

Shared source-aware reporting instructions:
- Use the diagnostic evidence table as the primary evidence source. Evaluate the public label rules against the task measurements and derived indicators before choosing the best-supported candidate label.
- Copy each evidence_id, source, field, statistic, direction, value, window, and role from its supplied observation. Keep diagnostic interpretation in supports, confidence, and limitations.
- The input evidence table has no hidden support labels. Infer supports from the observable evidence; do not invent hidden support annotations.
- Do not cite unavailable sources, fields, values, time windows, or evidence IDs. Do not overstate maintenance, metadata, or other contextual evidence as causal.
- Include the evidence needed to support the diagnosis or explain exclusions; do not omit grounded diagnostic evidence merely to shorten the report.
- If observations are insufficient or ambiguous, state the limitation and appropriate confidence instead of inventing evidence or forcing a fault or No Failure label.
- Recommend useful, specific next steps whose prerequisites are met. Every action must cite valid evidence IDs from the final evidence list.
- Follow the allowed action ontology. Shutdown and process-parameter adjustment are out of scope. Component replacement needs a target consistent with the predicted component and at least two valid non-contextual evidence items supporting that diagnosis.
- Sensor verification needs telemetry, machine_features, or derived_features evidence; error-log-only evidence is insufficient. Maintenance scheduling needs valid supporting evidence. Use inspection, monitoring, or further evidence collection when stronger actions are not justified.
- Select predicted_label from the candidate labels only and return a structured diagnostic report conforming to the required schema.
""".strip()


def build_m2_prompt(**kwargs) -> str:
    """M2 uses the shared initial prompt without runtime checking or repair."""
    return build_source_evidence_prompt(**kwargs)


def build_m3_prompt(**kwargs) -> str:
    """M3 has the exact M2 initial prompt; its additional controls run afterward."""
    return build_source_evidence_prompt(**kwargs)


# ============================================================
# Dispatcher
# ============================================================

def build_m3_repair_prompt(
    *,
    task: dict[str, Any],
    proxy_evidence: pd.DataFrame | list[dict[str, Any]],
    allowed_schema: dict[str, Any],
    action_ontology: dict[str, Any] | None,
    previous_report: dict[str, Any],
    violation_summary: str,
    association_guide: dict[str, Any] | None = None,
    runtime_feedback: dict[str, Any] | None = None,
    evidence_table_text: str | None = None,
) -> list[dict[str, str]]:
    """
    Build one-step repair prompt for M3.

    This prompt is used only by M3 API mode after deterministic verification
    detects schema, evidence, causal-overclaim, action-linkage, or unsafe-action
    violations.
    """
    system_prompt = build_system_prompt()

    user_prompt = f"""
Method: M3 one-step deterministic repair.

{format_common_contract()}

Task:
{format_task_context(task, include_auxiliary=True)}

Allowed diagnostic schema:
{format_allowed_schema(allowed_schema)}

Label decision guide:
{format_label_decision_guide(allowed_schema)}

Azure search-derived association guide:
{format_azure_association_guide(association_guide)}

Diagnostic evidence table:
{evidence_table_text if evidence_table_text is not None else format_proxy_evidence_table(proxy_evidence, max_rows=10)}

Action ontology:
{format_action_ontology(action_ontology)}

Previous structured report:
{previous_report}

Detected deterministic verification failures:
{violation_summary}

Structured observable corrections and reference records:
{format_runtime_corrections(runtime_feedback, proxy_evidence)}

{format_minimal_repair_contract()}

Repair instructions:
- Return a corrected structured diagnostic report.
- Correct each named field to the expected value from its observable record. Preserve the record's role even when its value is normal; do not change it to exclusionary.
- Keep predicted_label within the candidate labels only.
- Use only evidence available in the diagnostic evidence table.
- The input evidence table does not include supports; infer the supports field in the output report from the task context and candidate labels.
- Do not cite unavailable sources, fields, values, or time windows.
- Do not use contextual evidence as causal evidence.
- Remove or revise any action that lacks valid supporting evidence.
- Avoid out-of-scope actions such as shutdown or process parameter adjustment.
- Prefer conservative actions such as inspect, monitor, verify_sensor, collect_more_data, or escalate when evidence is limited.
- If a violation cannot be repaired without inventing evidence, state the limitation instead.
- If verify_sensor is recommended, it must be supported by telemetry, machine_features, or derived_features evidence. Do not use error-log-only evidence to justify verify_sensor.
- If only error-log evidence is available, prefer inspect, collect_more_data, or escalate instead of verify_sensor.
- Return only the corrected structured report.
""".strip()

    return [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]

def build_prompt_for_method(
    *,
    method: str,
    task: dict[str, Any],
    proxy_evidence: pd.DataFrame | list[dict[str, Any]] | None,
    allowed_schema: dict[str, Any],
    action_ontology: dict[str, Any] | None = None,
    association_guide: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    method = method.upper()

    system_prompt = build_system_prompt()

    if method == "M0":
        user_prompt = build_m0_prompt(
            task=task,
            allowed_schema=allowed_schema,
            action_ontology=action_ontology,
        )
    elif method == "M1":
        user_prompt = build_m1_prompt(
            task=task,
            allowed_schema=allowed_schema,
            action_ontology=action_ontology,
        )
    elif method == "M2":
        user_prompt = build_m2_prompt(
            task=task,
            proxy_evidence=proxy_evidence,
            allowed_schema=allowed_schema,
            action_ontology=action_ontology,
            association_guide=association_guide,
        )
    elif method == "M3":
        user_prompt = build_m3_prompt(
            task=task,
            proxy_evidence=proxy_evidence,
            allowed_schema=allowed_schema,
            action_ontology=action_ontology,
            association_guide=association_guide,
        )
    else:
        raise ValueError(f"Unsupported method for prompt builder: {method}")

    return [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]
