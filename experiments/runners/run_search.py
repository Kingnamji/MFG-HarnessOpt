"""
Search-set runner for seed harness evaluation.

Run from project root:

1. Dummy mode, no API calls:
    python -m experiments.runners.run_search --mode dummy

2. API mode, full configured search split:
    python -m experiments.runners.run_search --mode api

3. API mode, small pilot:
    python -m experiments.runners.run_search --mode api --ai4i-limit 5 --pdm-limit 5

This runner evaluates seed methods on the search split and saves traces for
future M4 harness optimization.

Default seed methods are loaded from:
    experiments/configs/experiment_config.yaml
    optimization.seed_methods: [M1, M2, M3]

Outputs:
    outputs/search/
    ├─ config_snapshot.json
    ├─ reports.jsonl
    ├─ traces.jsonl
    ├─ metrics_by_task.csv
    ├─ failure_traces.csv
    ├─ search_summary.csv
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
from experiments.harness.registry import create_harness
from experiments.llm.openai_client import OpenAIClient
from experiments.runners.aggregate_tables import aggregate_all_tables
from experiments.runners.run_smoke_test import (
    make_default_action_configs,
    run_api_method,
)
from experiments.utils.config_utils import (
    ExperimentConfigs,
    convert_paths_to_strings,
    get_api_key_path,
    get_output_dir,
    get_split_sizes,
    load_experiment_configs,
    validate_configs,
)

from experiments.runners.dataset_filter import parse_datasets_arg


# ============================================================
# I/O utilities
# ============================================================

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def parse_methods(methods_str: str | None, configs: ExperimentConfigs) -> list[str]:
    if methods_str:
        return [m.strip().upper() for m in methods_str.split(",") if m.strip()]

    seed_methods = configs.experiment.get("optimization", {}).get("seed_methods")
    if not seed_methods:
        seed_methods = ["M1", "M2", "M3"]

    return [str(m).upper() for m in seed_methods]


def make_output_dir(base_output_dir: Path, run_name: str | None) -> Path:
    if run_name:
        return base_output_dir / run_name
    return base_output_dir


# ============================================================
# Data selection
# ============================================================

def select_search_tasks(
    tasks: pd.DataFrame,
    proxy_evidence: pd.DataFrame,
    *,
    split: str,
    n: int,
    require_proxy_evidence: bool,
) -> pd.DataFrame:
    split_tasks = get_tasks_by_split(tasks, split)

    # The frozen task manifest defines the population, including empty observations.

    if len(split_tasks) < n:
        raise ValueError(
            f"Not enough tasks for split={split}, requested={n}, "
            f"available={len(split_tasks)}"
        )

    return split_tasks.head(n).reset_index(drop=True)


# ============================================================
# Client construction
# ============================================================

def create_client_from_configs(
    *,
    configs: ExperimentConfigs,
    mode: str,
    model_profile_name: str | None = None,
) -> OpenAIClient | None:
    if mode == "dummy":
        return None

    profile = configs.get_model_profile(model_profile_name or configs.model.get("diagnosis_model_profile"))

    return OpenAIClient(
        model=profile["model"],
        price_input_per_1m=float(profile.get("price_input_per_1m", 0.0)),
        price_output_per_1m=float(profile.get("price_output_per_1m", 0.0)),
        max_output_tokens=int(profile.get("max_output_tokens", 1200)),
        temperature=float(profile.get("temperature", 0.0)),
        dry_run=False,
        api_key_path=get_api_key_path(configs),
    )


# ============================================================
# Harness execution
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
):
    harness = create_harness(method, dummy_mode=True)

    return harness.run(
        task=task,
        proxy_evidence=proxy_evidence,
        allowed_schema=allowed_schema,
        action_ontology=action_ontology,
        action_preconditions=action_preconditions,
        generic_action_rules=generic_action_rules,
    )


def evaluate_harness_result(
    *,
    task: dict[str, Any],
    proxy_evidence: pd.DataFrame,
    allowed_schema: dict[str, Any],
    action_ontology: dict[str, Any],
    action_preconditions: dict[str, Any],
    generic_action_rules: dict[str, Any],
    harness_result,
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
    row["repair_failed"] = harness_result.repair_failed
    row["source_access_count"] = len(set(harness_result.accessed_sources))
    row["tool_call_count"] = len(harness_result.tool_calls)

    return row


# ============================================================
# Failure trace extraction
# ============================================================

def make_failure_trace_rows(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert metrics_by_task into long-form failure trace rows.

    Output columns:
    - task_id
    - dataset
    - method
    - harness_id
    - failure_id
    - failure_category
    - metric_context
    """
    failure_definitions = {
        "F1": "Label error",
        "F2": "Evidence omission",
        "F3": "Evidence contradiction",
        "F4": "Temporal violation",
        "F5": "Source hallucination",
        "F6": "Causal overclaim",
        "F7": "Action without evidence",
        "F8": "High-risk action violation",
        "F9": "Excessive source access",
        "F10": "Repair failure",
    }

    rows: list[dict[str, Any]] = []

    for _, row in metrics_df.iterrows():
        for flag, category in failure_definitions.items():
            if int(row.get(flag, 0) or 0) != 1:
                continue

            rows.append(
                {
                    "task_id": row.get("task_id"),
                    "dataset": row.get("dataset"),
                    "method": row.get("method"),
                    "harness_id": row.get("harness_id", ""),
                    "failure_id": flag,
                    "failure_category": category,
                    "gold_label": row.get("gold_label"),
                    "predicted_label": row.get("predicted_label"),
                    "evidence_precision": row.get("evidence_precision"),
                    "evidence_recall": row.get("evidence_recall"),
                    "unsupported_claim_rate": row.get("unsupported_claim_rate"),
                    "unsafe_recommendation_rate": row.get("unsafe_recommendation_rate"),
                    "generic_action_rate": row.get("generic_action_rate"),
                    "cost_per_report": row.get("cost_per_report"),
                    "repair_count": row.get("repair_count", 0),
                    "repair_failed": row.get("repair_failed", False),
                }
            )

    return pd.DataFrame(rows)


def make_search_summary(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compact method-level summary for search-set seed methods.
    """
    if metrics_df.empty:
        return pd.DataFrame()

    group_cols = ["method"]

    if "dataset" in metrics_df.columns:
        group_cols = ["dataset", "method"]

    agg_spec = {
        "label_correct": "mean",
        "evidence_precision": "mean",
        "evidence_recall": "mean",
        "unsupported_claim_rate": "mean",
        "unsafe_recommendation_rate": "mean",
        "generic_action_rate": "mean",
        "cost_per_report": "mean",
        "repair_count": "mean",
        "repair_failed": "mean",
        "source_access_count": "mean",
        "tool_call_count": "mean",
        "F1": "mean",
        "F2": "mean",
        "F3": "mean",
        "F4": "mean",
        "F5": "mean",
        "F6": "mean",
        "F7": "mean",
        "F8": "mean",
        "F9": "mean",
        "F10": "mean",
    }

    available_agg = {
        key: value for key, value in agg_spec.items()
        if key in metrics_df.columns
    }

    summary = (
        metrics_df
        .groupby(group_cols)
        .agg(available_agg)
        .reset_index()
    )

    return summary


# ============================================================
# Main runner
# ============================================================

def run_search(
    *,
    mode: str,
    configs: ExperimentConfigs,
    methods: list[str],
    split: str,
    ai4i_n: int,
    pdm_n: int,
    output_dir: Path,
    model_profile_name: str | None = None,
    require_proxy_evidence: bool = True,
    datasets_to_run: set[str] | None = None,
) -> None:
    mode = mode.lower().strip()
    datasets_to_run = datasets_to_run or {"AI4I", "PdM"}

    if mode not in {"dummy", "api"}:
        raise ValueError("mode must be one of: dummy, api")

    ensure_dir(output_dir)
    ensure_dir(output_dir / "tables")

    print(f"[INFO] Search runner mode: {mode}")
    print(f"[INFO] Split: {split}")
    print(f"[INFO] Methods: {methods}")
    print(f"[INFO] AI4I tasks: {ai4i_n}")
    print(f"[INFO] PdM tasks: {pdm_n}")
    print(f"[INFO] Output dir: {output_dir}")

    config_summary = validate_configs(configs)
    if not config_summary["valid"]:
        raise RuntimeError(f"Invalid config: {config_summary}")

    ai4i = load_ai4i_preprocessed()
    pdm = load_pdm_preprocessed()

    ai4i_action_ontology, ai4i_action_preconditions, ai4i_generic_rules = make_default_action_configs()

    ai4i_tasks = pd.DataFrame()
    pdm_tasks = pd.DataFrame()

    if "AI4I" in datasets_to_run:
        ai4i_tasks = select_search_tasks(
            ai4i.tasks,
            ai4i.generation_evidence,
            split=split,
            n=ai4i_n,
            require_proxy_evidence=require_proxy_evidence,
        )

    if "PdM" in datasets_to_run:
        pdm_tasks = select_search_tasks(
            pdm.tasks,
            pdm.generation_evidence,
            split=split,
            n=pdm_n,
            require_proxy_evidence=require_proxy_evidence,
        )



    client = create_client_from_configs(
        configs=configs,
        mode=mode,
        model_profile_name=model_profile_name,
    )

    reports_rows: list[dict[str, Any]] = []
    traces_rows: list[dict[str, Any]] = []
    metrics_rows: list[dict[str, Any]] = []

    datasets = []

    if "AI4I" in datasets_to_run:
        datasets.append(
            {
                "dataset_key": "AI4I",
                "name": "AI4I",
                "tasks": ai4i_tasks,
                "proxy_evidence": ai4i.proxy_evidence,
                "generation_evidence": ai4i.generation_evidence,
                "allowed_schema": ai4i.allowed_schema,
                "action_ontology": ai4i_action_ontology,
                "action_preconditions": ai4i_action_preconditions,
                "generic_action_rules": ai4i_generic_rules,
                "association_guide": {},
            }
        )

    if "PdM" in datasets_to_run:
        datasets.append(
            {
                "dataset_key": "PdM",
                "name": "PdM",
                "tasks": pdm_tasks,
                "proxy_evidence": pdm.proxy_evidence,
                "generation_evidence": pdm.generation_evidence,
                "allowed_schema": pdm.allowed_schema,
                "action_ontology": pdm.action_ontology,
                "action_preconditions": pdm.action_preconditions,
                "generic_action_rules": pdm.generic_action_rules,
                "association_guide": pdm.association_guide,
            }
        )

    total_calls = len(methods) * sum(len(bundle["tasks"]) for bundle in datasets)
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
                    if client is None:
                        raise RuntimeError("API mode requires OpenAIClient.")

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

                metric_row = evaluate_harness_result(
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
                        "split": split,
                        "method": method,
                        "harness_id": harness_result.harness_id,
                        "report": harness_result.report,
                    }
                )

                traces_rows.append(
                    {
                        "task_id": task_id,
                        "dataset": task.get("dataset", dataset_name),
                        "split": split,
                        "method": method,
                        "harness_id": harness_result.harness_id,
                        "execution_log": harness_result.execution_log(),
                        "metric_row": metric_row,
                        "raw_response": harness_result.raw_response,
                    }
                )

                metrics_rows.append(metric_row)

    metrics_df = pd.DataFrame(metrics_rows)
    failure_traces_df = make_failure_trace_rows(metrics_df)
    search_summary_df = make_search_summary(metrics_df)

    reports_path = output_dir / "reports.jsonl"
    traces_path = output_dir / "traces.jsonl"
    metrics_path = output_dir / "metrics_by_task.csv"
    failure_traces_path = output_dir / "failure_traces.csv"
    search_summary_path = output_dir / "search_summary.csv"
    config_snapshot_path = output_dir / "config_snapshot.json"

    write_jsonl(reports_path, reports_rows)
    write_jsonl(traces_path, traces_rows)
    metrics_df.to_csv(metrics_path, index=False)
    failure_traces_df.to_csv(failure_traces_path, index=False)
    search_summary_df.to_csv(search_summary_path, index=False)

    config_snapshot = {
        "mode": mode,
        "split": split,
        "datasets": sorted(datasets_to_run),
        "methods": methods,
        "ai4i_n": ai4i_n,
        "pdm_n": pdm_n,
        "output_dir": str(output_dir),
        "config_summary": config_summary,
        "paths": convert_paths_to_strings(configs.paths),
        "model": configs.model,
        "experiment": configs.experiment,
        "metric_weights": configs.metric_weights,
        "methods_config": configs.methods,
        "association_guide": {
            "pdm_loaded": bool(getattr(pdm, "association_guide", {})),
            "pdm_top_associations": len(
                getattr(pdm, "association_guide", {}).get("top_associations", [])
            ) if getattr(pdm, "association_guide", {}) else 0,
        },
    }
    write_json(config_snapshot_path, config_snapshot)

    print(f"[INFO] Saved reports: {reports_path}")
    print(f"[INFO] Saved traces: {traces_path}")
    print(f"[INFO] Saved metrics: {metrics_path} {metrics_df.shape}")
    print(f"[INFO] Saved failure traces: {failure_traces_path} {failure_traces_df.shape}")
    print(f"[INFO] Saved search summary: {search_summary_path} {search_summary_df.shape}")
    print(f"[INFO] Saved config snapshot: {config_snapshot_path}")

    aggregate_all_tables(
        test_metrics_path=metrics_path,
        output_dir=output_dir / "tables",
        search_metrics_path=None,
        digits=4,
    )

    print(f"[INFO] Saved search tables: {output_dir / 'tables'}")

    print("\n=== Search summary ===")
    if search_summary_df.empty:
        print("No search summary generated.")
    else:
        print(search_summary_df)

    print("\nSearch run completed.")


# ============================================================
# CLI
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Run search-set seed harness evaluation.")

    parser.add_argument(
        "--mode",
        choices=["dummy", "api"],
        default="dummy",
        help="dummy = no API call; api = actual OpenAI API call.",
    )
    parser.add_argument(
    "--datasets",
    default="AI4I,PdM",
    help="Comma-separated dataset filter: AI4I, PdM, or AI4I,PdM.",
    )
    parser.add_argument(
        "--methods",
        default=None,
        help="Comma-separated methods. Default: optimization.seed_methods from config.",
    )
    parser.add_argument(
        "--split",
        default=None,
        help="Dataset split. Default: optimization.search_split from config.",
    )
    parser.add_argument(
        "--ai4i-limit",
        type=int,
        default=None,
        help="Optional override for number of AI4I search tasks.",
    )
    parser.add_argument(
        "--pdm-limit",
        type=int,
        default=None,
        help="Optional override for number of PdM search tasks.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Default: paths.outputs.search_dir.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional subdirectory under output-dir.",
    )
    parser.add_argument(
        "--model-profile",
        default=None,
        help="Optional model profile name from model_config.yaml.",
    )
    parser.add_argument(
        "--allow-tasks-without-evidence",
        action="store_true",
        help="If set, do not require proxy evidence when selecting tasks.",
    )

    args = parser.parse_args()

    configs = load_experiment_configs()

    methods = parse_methods(args.methods, configs)

    split = args.split or configs.experiment.get("optimization", {}).get("search_split", "search")
    split_sizes = get_split_sizes(configs, split)

    ai4i_n = args.ai4i_limit if args.ai4i_limit is not None else int(split_sizes["ai4i_n"])
    pdm_n = args.pdm_limit if args.pdm_limit is not None else int(split_sizes["pdm_n"])

    base_output_dir = Path(args.output_dir) if args.output_dir else get_output_dir(configs, "search_dir")
    output_dir = make_output_dir(base_output_dir, args.run_name)

    datasets_to_run = parse_datasets_arg(args.datasets)

    run_search(
        mode=args.mode,
        configs=configs,
        methods=methods,
        split=split,
        ai4i_n=ai4i_n,
        pdm_n=pdm_n,
        output_dir=output_dir,
        model_profile_name=args.model_profile,
        require_proxy_evidence=not args.allow_tasks_without_evidence,
        datasets_to_run=datasets_to_run,
    )


if __name__ == "__main__":
    main()