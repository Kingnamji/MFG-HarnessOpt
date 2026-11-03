"""
Smoke test runner for M0--M3 diagnostic harnesses.

Run from project root:

1. Dummy mode, no API call:
    python -m experiments.runners.run_smoke_test --mode dummy

2. API mode, actual OpenAI call:
    python -m experiments.runners.run_smoke_test --mode api --model gpt-4.1-mini

The runner:
- loads preprocessed AI4I and Azure PdM tasks
- selects small search-set subsets
- runs M0--M3
- evaluates each report using deterministic evaluator
- saves reports, traces, metrics_by_task.csv
- generates Table 6--11 style smoke tables

Outputs:
    outputs/smoke_test/
    ├─ reports.jsonl
    ├─ traces.jsonl
    ├─ metrics_by_task.csv
    └─ tables/
       ├─ table6_overall.csv
       ├─ table7_datasetwise.csv
       ├─ table8_component_effects.csv
       ├─ table9_optimization_trajectory.csv
       ├─ table10_failure_trace_analysis.csv
       └─ table11_reliability_cost.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from dataclasses import dataclass

from experiments.data.load_preprocessed import (
    get_proxy_evidence_for_task,
    get_tasks_by_split,
    load_ai4i_preprocessed,
    load_pdm_preprocessed,
)
from experiments.evaluation.deterministic_evaluator import (
    evaluate_report,
    evaluation_result_to_dict,
)
from experiments.harness.base import HarnessResult
from experiments.harness.registry import create_harness
from experiments.llm.openai_client import OpenAIClient
from experiments.llm.prompts import build_prompt_for_method, build_m3_repair_prompt
from experiments.evaluation.runtime_verifier import verify_runtime_report, runtime_failure_summary, require_runtime_result
from experiments.data.generation_evidence import validate_generation_evidence, validate_observation_windows
from experiments.utils.provenance import runtime_provenance
from experiments.runners.aggregate_tables import aggregate_all_tables


# ============================================================
# Defaults
# ============================================================

DEFAULT_METHODS = ["M0", "M1", "M2", "M3"]

DEFAULT_OUTPUT_DIR = Path("outputs") / "smoke_test"

@dataclass(frozen=True)
class LLMResultLike:
    parsed_report: dict[str, Any]
    raw_response: Any
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_sec: float
    model_name: str


# ============================================================
# Shared action configs
# ============================================================

def make_default_action_configs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """
    Shared action configs used for AI4I.

    Azure PdM has these files in preprocessed/PdM, but AI4I currently stores
    only dataset schema/config. For consistency, use the same ontology.
    """
    action_ontology = {
        "action_types": {
            "monitor": {"risk": "low", "allowed": True},
            "inspect": {"risk": "low", "allowed": True},
            "verify_sensor": {"risk": "low_medium", "allowed": True},
            "collect_more_data": {"risk": "low", "allowed": True},
            "escalate": {"risk": "low", "allowed": True},
            "schedule_maintenance": {"risk": "medium", "allowed": True},
            "replace_component": {"risk": "high", "allowed": True},
            "adjust_parameter": {"risk": "out_of_scope", "allowed": False},
            "shutdown": {"risk": "out_of_scope", "allowed": False},
        }
    }

    action_preconditions = {
        "monitor": {"requires": ["valid_schema"]},
        "inspect": {"requires": ["target_present"]},
        "verify_sensor": {"requires": ["valid_telemetry_or_sensor_evidence"]},
        "collect_more_data": {"requires": ["target_or_missing_evidence_reason"]},
        "escalate": {"requires": ["valid_schema"]},
        "schedule_maintenance": {"requires": ["at_least_one_valid_evidence_item"]},
        "replace_component": {
            "requires": [
                "component_specific_predicted_label",
                "target_matches_predicted_label",
                "at_least_two_valid_supporting_evidence_items",
            ]
        },
        "adjust_parameter": {"always_unsafe": True},
        "shutdown": {"always_unsafe": True},
    }

    generic_action_rules = {
        "generic_if": {
            "action_type_in": ["monitor", "escalate", "collect_more_data"],
            "target_in_or_missing": [
                "",
                "system",
                "machine",
                "equipment",
                "maintenance team",
                "operator",
            ],
            "requires_no_valid_rationale_evidence": True,
        }
    }

    return action_ontology, action_preconditions, generic_action_rules


# ============================================================
# I/O utilities
# ============================================================

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def parse_methods(methods_str: str) -> list[str]:
    methods = [m.strip().upper() for m in methods_str.split(",") if m.strip()]
    invalid = [m for m in methods if m not in DEFAULT_METHODS]

    if invalid:
        raise ValueError(f"Unsupported methods: {invalid}. Use subset of {DEFAULT_METHODS}.")

    return methods


def select_smoke_tasks(
    tasks: pd.DataFrame,
    proxy_evidence: pd.DataFrame,
    *,
    split: str,
    n: int,
    require_evidence: bool = True,
) -> pd.DataFrame:
    split_tasks = get_tasks_by_split(tasks, split)

    # The frozen task manifest defines the population, including empty observations.

    if len(split_tasks) < n:
        raise ValueError(
            f"Not enough tasks for split={split}, n={n}. "
            f"Available={len(split_tasks)}."
        )

    return split_tasks.head(n).reset_index(drop=True)


# ============================================================
# API-mode harness execution
# ============================================================

def infer_accessed_sources_for_method(
    *,
    method: str,
    proxy_evidence: pd.DataFrame,
) -> list[str]:
    method = method.upper()


    if method == "M0":
        return []

    if method == "M1":
        return ["static_summary"]

    if proxy_evidence.empty or "source" not in proxy_evidence.columns:
        return []

    if method in {"M2", "M3"}:
        return sorted(set(proxy_evidence["source"].dropna().astype(str).tolist()))

    return []


def infer_tool_calls_for_method(method: str) -> list[dict[str, Any]]:
    method = method.upper()

    if method == "M0":
        return []

    if method == "M1":
        return [
            {"tool_name": "build_static_summary", "status": "ok"},
        ]

    if method == "M2":
        return [
            {"tool_name": "source_policy_lookup", "status": "ok"},
            {"tool_name": "diagnostic_evidence_table", "status": "ok"},
        ]

    if method == "M3":
        return [
            {"tool_name": "source_policy_lookup", "status": "ok"},
            {"tool_name": "diagnostic_evidence_table", "status": "ok"},
            {"tool_name": "deterministic_verifier", "status": "ok"},
            {"tool_name": "action_validity_filter", "status": "ok"},
        ]

    return []


def run_api_method(
    *,
    method: str,
    task: dict[str, Any],
    proxy_evidence: pd.DataFrame,
    allowed_schema: dict[str, Any],
    action_ontology: dict[str, Any],
    action_preconditions: dict[str, Any] | None,
    generic_action_rules: dict[str, Any] | None,
    client: OpenAIClient,
    association_guide: dict[str, Any] | None = None,
) -> HarnessResult:
    """
    Run one method using OpenAI API.

    M0--M2:
    - one LLM call

    M3:
    - first LLM call
    - deterministic evaluation
    - one repair call if fixed verifier detects repair-triggering failures
    """
    method = method.upper()

    # The legacy argument name is retained for runner compatibility; the API
    # boundary accepts only the separate observation table, never target data.
    validate_generation_evidence(proxy_evidence, [task["task_id"]])
    validate_observation_windows(proxy_evidence, pd.DataFrame([task]))
    messages = build_prompt_for_method(
        method=method,
        task=task,
        proxy_evidence=proxy_evidence,
        allowed_schema=allowed_schema,
        action_ontology=action_ontology,
        association_guide=association_guide,
    )

    first_result = client.generate_structured_report(messages=messages)

    accessed_sources = infer_accessed_sources_for_method(
        method=method,
        proxy_evidence=proxy_evidence,
    )
    tool_calls = infer_tool_calls_for_method(method)

    repair_count = 0
    repair_failed = False
    final_llm_result = first_result
    final_report = first_result.parsed_report

    # M3 deterministic verification + one-step repair.
    if method == "M3":
        provisional_harness_result = HarnessResult(
            method=method,
            harness_id=f"{method}_api_smoke",
            task_id=str(task.get("task_id", "")),
            dataset=str(task.get("dataset", "")),
            report=first_result.parsed_report,
            accessed_sources=accessed_sources,
            tool_calls=tool_calls,
            repair_count=0,
            repair_failed=False,
            latency_sec=first_result.latency_sec,
            input_tokens=first_result.input_tokens,
            output_tokens=first_result.output_tokens,
            cost_usd=first_result.cost_usd,
            model_name=first_result.model_name,
            raw_response=first_result.raw_response,
        )

        provisional_metric_row = verify_runtime_report(
            report=first_result.parsed_report,
            generation_evidence=proxy_evidence,
            allowed_schema=allowed_schema,
            action_ontology=action_ontology,
            action_preconditions=action_preconditions or {},
            generic_action_rules=generic_action_rules or {},
        )

        if should_repair_m3(provisional_metric_row):
            violation_summary = build_violation_summary_from_metric_row(provisional_metric_row)

            repair_messages = build_m3_repair_prompt(
                task=task,
                proxy_evidence=proxy_evidence,
                allowed_schema=allowed_schema,
                action_ontology=action_ontology,
                previous_report=first_result.parsed_report,
                violation_summary=violation_summary,
                runtime_feedback=provisional_metric_row,
                association_guide=association_guide,
            )

            repair_result = client.generate_structured_report(messages=repair_messages)

            repair_count = 1
            final_llm_result = repair_result
            final_report = repair_result.parsed_report

            # Check whether repair still fails.
            repaired_harness_result = HarnessResult(
                method=method,
                harness_id=f"{method}_api_smoke",
                task_id=str(task.get("task_id", "")),
                dataset=str(task.get("dataset", "")),
                report=final_report,
                accessed_sources=accessed_sources,
                tool_calls=tool_calls + [
                    {
                        "tool_name": "m3_one_step_repair",
                        "status": "ok",
                    }
                ],
                repair_count=repair_count,
                repair_failed=False,
                latency_sec=first_result.latency_sec + repair_result.latency_sec,
                input_tokens=first_result.input_tokens + repair_result.input_tokens,
                output_tokens=first_result.output_tokens + repair_result.output_tokens,
                cost_usd=first_result.cost_usd + repair_result.cost_usd,
                model_name=repair_result.model_name,
                raw_response={
                    "first_response": first_result.raw_response,
                    "repair_response": repair_result.raw_response,
                    "violation_summary": violation_summary,
                },
            )

            repaired_metric_row = verify_runtime_report(
                report=final_report,
                generation_evidence=proxy_evidence,
                allowed_schema=allowed_schema,
                action_ontology=action_ontology,
                action_preconditions=action_preconditions or {},
                generic_action_rules=generic_action_rules or {},
            )

            repair_failed = should_repair_m3(repaired_metric_row)

            final_llm_result = LLMResultLike(
                parsed_report=final_report,
                raw_response={
                    "first_response": first_result.raw_response,
                    "repair_response": repair_result.raw_response,
                    "violation_summary": violation_summary,
                },
                input_tokens=first_result.input_tokens + repair_result.input_tokens,
                output_tokens=first_result.output_tokens + repair_result.output_tokens,
                cost_usd=first_result.cost_usd + repair_result.cost_usd,
                latency_sec=first_result.latency_sec + repair_result.latency_sec,
                model_name=repair_result.model_name,
            )

    from experiments.evaluation.runtime_verifier import audit_runtime_repair
    postprocessing = None
    pre_filter_report = final_report
    repair_attempt_failed = None
    if method == "M3":
        from experiments.harness.verified_postprocessing import finalize_verified_report
        postprocessing = finalize_verified_report(
            report=pre_filter_report, generation_evidence=proxy_evidence,
            allowed_schema=allowed_schema, action_ontology=action_ontology,
            action_preconditions=action_preconditions, generic_action_rules=generic_action_rules,
        )
        repair_attempt_failed = should_repair_m3(postprocessing['pre_filter_verification']) if repair_count else None
        final_report = postprocessing['report']
        repair_failed = bool(repair_count and should_repair_m3(postprocessing['final_runtime_verification']))
    request_log = {
        "responses": final_llm_result.raw_response, "initial_messages": messages,
        "initial_report": first_result.parsed_report,
        "repair_report": pre_filter_report if repair_count else None,
        "pre_filter_report": pre_filter_report,
        "action_filter": postprocessing['action_filter'] if postprocessing else None,
        "pre_filter_runtime_verification": postprocessing['pre_filter_verification'] if postprocessing else None,
        "repair_attempt_failed": repair_attempt_failed,
        "repair_messages": repair_messages if repair_count else None,
        "initial_runtime_verification": provisional_metric_row if method == "M3" else None,
        "final_runtime_verification": postprocessing['final_runtime_verification'] if postprocessing else None,
        "repair_metadata_audit": audit_runtime_repair(first_result.parsed_report,pre_filter_report,proxy_evidence) if repair_count else None,
    }
    return HarnessResult(
        method=method,
        harness_id=f"{method}_api_smoke",
        task_id=str(task.get("task_id", "")),
        dataset=str(task.get("dataset", "")),
        report=final_report,
        accessed_sources=accessed_sources,
        tool_calls=tool_calls + (
            [{"tool_name": "m3_one_step_repair", "status": "ok"}]
            if method == "M3" and repair_count > 0
            else []
        ),
        repair_count=repair_count,
        repair_failed=repair_failed,
        latency_sec=final_llm_result.latency_sec,
        input_tokens=final_llm_result.input_tokens,
        output_tokens=final_llm_result.output_tokens,
        cost_usd=final_llm_result.cost_usd,
        model_name=final_llm_result.model_name,
        raw_response=request_log,
        extra={"protocol": runtime_provenance(), "action_filter": postprocessing['action_filter'] if postprocessing else None,
               "repair_attempt_failed": repair_attempt_failed},
    )


# ============================================================
# Dummy-mode harness execution
# ============================================================

def run_dummy_method(
    *,
    method: str,
    task: dict[str, Any],
    proxy_evidence: pd.DataFrame,
    allowed_schema: dict[str, Any],
    action_ontology: dict[str, Any],
    action_preconditions: dict[str, Any],
    generic_action_rules: dict[str, Any],
) -> HarnessResult:
    harness = create_harness(method, dummy_mode=True)

    return harness.run(
        task=task,
        proxy_evidence=proxy_evidence,
        allowed_schema=allowed_schema,
        action_ontology=action_ontology,
        action_preconditions=action_preconditions,
        generic_action_rules=generic_action_rules,
    )


# ============================================================
# Evaluation wrapper
# ============================================================

def build_violation_summary_from_metric_row(metric_row: dict[str, Any]) -> str:
    """Return observable-only repair feedback."""
    return runtime_failure_summary(metric_row)

def should_repair_m3(metric_row: dict[str, Any]) -> bool:
    """
    Fixed M3 repair trigger.

    This is intentionally simple and frozen:
    - schema invalid
    - F5 source hallucination
    - F6 causal overclaim
    - F7 action without evidence
    - F8 high-risk action violation
    - unsafe recommendation rate > 0
    """
    require_runtime_result(metric_row)
    if metric_row.get('public_condition_consistency', {}).get('errors'):
        return True
    if metric_row.get("schema_valid", 1) == 0 or metric_row.get("evidence_errors"):
        return True

    for flag in ["F3", "F4", "F5", "F6", "F7", "F8"]:
        if int(metric_row.get(flag, 0) or 0) == 1:
            return True

    if float(metric_row.get("unsafe_recommendation_rate", 0.0) or 0.0) > 0:
        return True

    return False


def evaluate_harness_output(
    *,
    task: dict[str, Any],
    proxy_evidence: pd.DataFrame,
    allowed_schema: dict[str, Any],
    action_ontology: dict[str, Any],
    action_preconditions: dict[str, Any],
    generic_action_rules: dict[str, Any],
    harness_result: HarnessResult,
) -> dict[str, Any]:
    evaluation = evaluate_report(
        task=task,
        report=harness_result.report,
        proxy_evidence=proxy_evidence,
        allowed_schema=allowed_schema,
        action_ontology=action_ontology,
        action_preconditions=action_preconditions,
        generic_action_rules=generic_action_rules,
        execution_log=harness_result.execution_log(),
        method=harness_result.method,
        repair_failed=harness_result.repair_failed,
    )

    row = evaluation_result_to_dict(evaluation)

    row["harness_id"] = harness_result.harness_id
    row["model_name"] = harness_result.model_name
    row["input_tokens"] = harness_result.input_tokens
    row["output_tokens"] = harness_result.output_tokens
    row["latency_sec"] = harness_result.latency_sec
    row["repair_count"] = harness_result.repair_count
    row["source_access_count"] = len(set(harness_result.accessed_sources))
    row["tool_call_count"] = len(harness_result.tool_calls)

    return row


# ============================================================
# Main smoke runner
# ============================================================

def run_smoke_test(
    *,
    mode: str,
    methods: list[str],
    split: str,
    ai4i_n: int,
    pdm_n: int,
    output_dir: Path,
    model: str,
    price_input_per_1m: float,
    price_output_per_1m: float,
    max_output_tokens: int,
    temperature: float,
    api_key_path: str | None,
) -> None:
    mode = mode.lower().strip()
    if mode not in {"dummy", "api"}:
        raise ValueError("mode must be one of: dummy, api")

    ensure_dir(output_dir)
    ensure_dir(output_dir / "tables")

    print(f"[INFO] Smoke test mode: {mode}")
    print(f"[INFO] Methods: {methods}")
    print(f"[INFO] Output dir: {output_dir}")

    ai4i = load_ai4i_preprocessed()
    pdm = load_pdm_preprocessed()

    ai4i_action_ontology, ai4i_action_preconditions, ai4i_generic_rules = make_default_action_configs()

    ai4i_tasks = select_smoke_tasks(
        ai4i.tasks,
        ai4i.proxy_evidence,
        split=split,
        n=ai4i_n,
        require_evidence=True,
    )
    pdm_tasks = select_smoke_tasks(
        pdm.tasks,
        pdm.proxy_evidence,
        split=split,
        n=pdm_n,
        require_evidence=True,
    )

    client: OpenAIClient | None = None
    if mode == "api":
        client = OpenAIClient(
            model=model,
            price_input_per_1m=price_input_per_1m,
            price_output_per_1m=price_output_per_1m,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            dry_run=False,
            api_key_path=api_key_path
        )

    reports_rows: list[dict[str, Any]] = []
    traces_rows: list[dict[str, Any]] = []
    metrics_rows: list[dict[str, Any]] = []

    datasets = [
        {
            "name": "AI4I",
            "tasks": ai4i_tasks,
            "proxy_evidence": ai4i.proxy_evidence,
            "generation_evidence": ai4i.generation_evidence,
            "allowed_schema": ai4i.allowed_schema,
            "action_ontology": ai4i_action_ontology,
            "action_preconditions": ai4i_action_preconditions,
            "generic_action_rules": ai4i_generic_rules,
            "association_guide": {},
        },
        {
            "name": "PdM",
            "tasks": pdm_tasks,
            "proxy_evidence": pdm.proxy_evidence,
            "generation_evidence": pdm.generation_evidence,
            "allowed_schema": pdm.allowed_schema,
            "action_ontology": pdm.action_ontology,
            "action_preconditions": pdm.action_preconditions,
            "generic_action_rules": pdm.generic_action_rules,
            "association_guide": pdm.association_guide,
        },
    ]

    total_calls = len(methods) * (len(ai4i_tasks) + len(pdm_tasks))
    call_idx = 0

    for dataset_bundle in datasets:
        dataset_name = dataset_bundle["name"]
        tasks_df = dataset_bundle["tasks"]
        proxy_all = dataset_bundle["proxy_evidence"]

        for _, task_row in tasks_df.iterrows():
            task = task_row.to_dict()
            task_id = str(task["task_id"])
            proxy = get_proxy_evidence_for_task(proxy_all, task_id)
            generation = get_proxy_evidence_for_task(dataset_bundle["generation_evidence"], task_id)

            for method in methods:
                call_idx += 1
                print(f"[INFO] ({call_idx}/{total_calls}) {dataset_name} {task_id} {method}")

                if mode == "dummy":
                    harness_result = run_dummy_method(
                        method=method,
                        task=task,
                        proxy_evidence=generation,
                        allowed_schema=dataset_bundle["allowed_schema"],
                        action_ontology=dataset_bundle["action_ontology"],
                        action_preconditions=dataset_bundle["action_preconditions"],
                        generic_action_rules=dataset_bundle["generic_action_rules"],
                    )
                else:
                    assert client is not None
                    harness_result = run_api_method(
                        method=method,
                        task=task,
                        proxy_evidence=generation,
                        allowed_schema=dataset_bundle["allowed_schema"],
                        action_ontology=dataset_bundle["action_ontology"],
                        action_preconditions=dataset_bundle["action_preconditions"],
                        generic_action_rules=dataset_bundle["generic_action_rules"],
                        association_guide=dataset_bundle["association_guide"],
                        client=client,
                    )

                metric_row = evaluate_harness_output(
                    task=task,
                    proxy_evidence=proxy,
                    allowed_schema=dataset_bundle["allowed_schema"],
                    action_ontology=dataset_bundle["action_ontology"],
                    action_preconditions=dataset_bundle["action_preconditions"],
                    generic_action_rules=dataset_bundle["generic_action_rules"],
                    harness_result=harness_result,
                )

                reports_rows.append(
                    {
                        "task_id": task_id,
                        "dataset": task.get("dataset", dataset_name),
                        "method": method,
                        "harness_id": harness_result.harness_id,
                        "report": harness_result.report,
                    }
                )

                traces_rows.append(
                    {
                        "task_id": task_id,
                        "dataset": task.get("dataset", dataset_name),
                        "method": method,
                        "harness_id": harness_result.harness_id,
                        "execution_log": harness_result.execution_log(),
                        "raw_response": harness_result.raw_response,
                    }
                )

                metrics_rows.append(metric_row)

    metrics_df = pd.DataFrame(metrics_rows)

    reports_path = output_dir / "reports.jsonl"
    traces_path = output_dir / "traces.jsonl"
    metrics_path = output_dir / "metrics_by_task.csv"

    write_jsonl(reports_path, reports_rows)
    write_jsonl(traces_path, traces_rows)
    metrics_df.to_csv(metrics_path, index=False)

    print(f"[INFO] Saved reports: {reports_path}")
    print(f"[INFO] Saved traces: {traces_path}")
    print(f"[INFO] Saved metrics: {metrics_path} {metrics_df.shape}")

    aggregate_all_tables(
        test_metrics_path=metrics_path,
        output_dir=output_dir / "tables",
        search_metrics_path=None,
        digits=4,
    )

    print(f"[INFO] Saved smoke tables: {output_dir / 'tables'}")

    print("\n=== Smoke test summary ===")
    summary_cols = [
        "method",
        "dataset",
        "label_correct",
        "evidence_precision",
        "evidence_recall",
        "unsupported_claim_rate",
        "unsafe_recommendation_rate",
        "generic_action_rate",
        "cost_per_report",
    ]

    available_cols = [col for col in summary_cols if col in metrics_df.columns]
    print(metrics_df[available_cols].groupby(["dataset", "method"]).mean(numeric_only=True))

    print("\nSmoke test completed.")


# ============================================================
# CLI
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Run M0--M3 smoke test.")

    parser.add_argument(
        "--api-key-path",
        default=None,
        help="Path to JSON file containing OPENAI_API_KEY.",
    )

    parser.add_argument(
        "--mode",
        choices=["dummy", "api"],
        default="dummy",
        help="dummy = no API call; api = actual OpenAI API call.",
    )
    parser.add_argument(
        "--methods",
        default="M0,M1,M2,M3",
        help="Comma-separated methods to run. Default: M0,M1,M2,M3.",
    )
    parser.add_argument(
        "--split",
        default="search",
        help="Dataset split to sample from. Default: search.",
    )
    parser.add_argument(
        "--ai4i-n",
        type=int,
        default=5,
        help="Number of AI4I tasks for smoke test.",
    )
    parser.add_argument(
        "--pdm-n",
        type=int,
        default=5,
        help="Number of Azure PdM tasks for smoke test.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory.",
    )
    parser.add_argument(
        "--model",
        default="gpt-4.1-mini",
        help="OpenAI model for api mode.",
    )
    parser.add_argument(
        "--price-input-per-1m",
        type=float,
        default=0.0,
        help="Input token price per 1M tokens for cost logging.",
    )
    parser.add_argument(
        "--price-output-per-1m",
        type=float,
        default=0.0,
        help="Output token price per 1M tokens for cost logging.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=1200,
        help="Maximum output tokens for API call.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Generation temperature.",
    )

    args = parser.parse_args()

    methods = parse_methods(args.methods)

    run_smoke_test(
        mode=args.mode,
        methods=methods,
        split=args.split,
        ai4i_n=args.ai4i_n,
        pdm_n=args.pdm_n,
        output_dir=Path(args.output_dir),
        model=args.model,
        price_input_per_1m=args.price_input_per_1m,
        price_output_per_1m=args.price_output_per_1m,
        max_output_tokens=args.max_output_tokens,
        temperature=args.temperature,
        api_key_path=args.api_key_path,
    )


if __name__ == "__main__":
    main()
