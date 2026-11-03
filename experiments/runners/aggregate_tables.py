"""
Aggregate task-level metrics into paper tables.

This module generates the planned Table 6--11 outputs from task-level
metrics CSV files.

Expected task-level metrics columns
-----------------------------------
Required for final test aggregation:
- task_id
- dataset
- method
- gold_label
- predicted_label
- evidence_precision
- evidence_recall
- unsupported_claim_rate
- unsafe_recommendation_rate
- generic_action_rate
- cost_per_report
- source_hallucination_rate
- temporal_violation_rate
- evidence_linked_action_ratio
- causal_overclaim_rate
- F1 ... F10

Optional for optimization trajectory:
- stage
- candidate_id
- J_mfg

Outputs
-------
- table6_overall.csv
- table7_datasetwise.csv
- table8_component_effects.csv
- table9_optimization_trajectory.csv
- table10_failure_trace_analysis.csv
- table11_reliability_cost.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


# ============================================================
# Constants
# ============================================================

METHOD_ORDER = ["M0", "M1", "M2", "M3", "M4"]

MAIN_METRICS = [
    "macro_f1",
    "evidence_precision",
    "evidence_recall",
    "unsupported_claim_rate",
    "unsafe_recommendation_rate",
    "generic_action_rate",
    "cost_per_report",
]

DATASETWISE_METRICS = [
    "macro_f1",
    "evidence_precision",
    "evidence_recall",
    "unsupported_claim_rate",
    "unsafe_recommendation_rate",
    "generic_action_rate",
    "cost_per_report",
]

COMPONENT_METRICS = [
    "evidence_recall",
    "source_hallucination_rate",
    "temporal_violation_rate",
    "evidence_linked_action_ratio",
    "unsafe_recommendation_rate",
    "causal_overclaim_rate",
]

OPTIMIZATION_METRICS = [
    "J_mfg",
    "macro_f1",
    "evidence_precision",
    "unsupported_claim_rate",
    "unsafe_recommendation_rate",
    "cost_per_report",
]

RELIABILITY_COST_METRICS = [
    "macro_f1",
    "evidence_precision",
    "unsupported_claim_rate",
    "unsafe_recommendation_rate",
    "cost_per_report",
]

FAILURE_FLAGS = [f"F{i}" for i in range(1, 11)]


# ============================================================
# Utilities
# ============================================================

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_metrics_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Metrics file not found: {path}")

    df = pd.read_csv(path)
    return normalize_metric_columns(df)


def normalize_metric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names and method labels for robust aggregation.
    """
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    rename_map = {
        "Evidence P": "evidence_precision",
        "Evidence R": "evidence_recall",
        "Unsupported": "unsupported_claim_rate",
        "Unsafe": "unsafe_recommendation_rate",
        "Generic Action": "generic_action_rate",
        "Cost/report": "cost_per_report",
        "Source Hallucination": "source_hallucination_rate",
        "Temporal Violation": "temporal_violation_rate",
        "Evidence-linked Action": "evidence_linked_action_ratio",
        "Causal Overclaim": "causal_overclaim_rate",
        "Macro-F1": "macro_f1",
        "J": "J_mfg",
    }

    df = df.rename(columns=rename_map)

    if "method" in df.columns:
        df["method"] = df["method"].astype(str)

    if "dataset" in df.columns:
        df["dataset"] = df["dataset"].astype(str)

    return df


def require_columns(df: pd.DataFrame, columns: Iterable[str], table_name: str) -> None:
    missing = [col for col in columns if col not in df.columns]

    if missing:
        raise KeyError(
            f"{table_name} requires missing columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )


def safe_mean(series: pd.Series) -> float:
    """
    Mean that ignores NaN/None. Returns np.nan if no valid values exist.
    """
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.dropna().empty:
        return np.nan
    return float(numeric.mean())


def ordered_methods(methods: Iterable[str]) -> list[str]:
    methods = list(dict.fromkeys(str(m) for m in methods))
    ordered = [m for m in METHOD_ORDER if m in methods]
    ordered.extend([m for m in methods if m not in ordered])
    return ordered


def compute_macro_f1(group: pd.DataFrame) -> float:
    require_columns(group, ["gold_label", "predicted_label"], "macro_f1")

    y_true = group["gold_label"].astype(str)
    y_pred = group["predicted_label"].astype(str)

    labels = sorted(set(y_true) | set(y_pred))
    if len(labels) == 0:
        return np.nan

    return float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))


def aggregate_group_metrics(
    group: pd.DataFrame,
    metric_names: list[str],
) -> dict[str, float]:
    """
    Aggregate one group of task-level metrics.

    macro_f1 is computed from gold/predicted labels.
    Other metrics are averaged at the task level.
    """
    row: dict[str, float] = {}

    for metric in metric_names:
        if metric == "macro_f1":
            row[metric] = compute_macro_f1(group)
        else:
            if metric not in group.columns:
                row[metric] = np.nan
            else:
                row[metric] = safe_mean(group[metric])

    return row


def round_table(df: pd.DataFrame, digits: int = 4) -> pd.DataFrame:
    out = df.copy()

    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].round(digits)

    return out


def write_table(df: pd.DataFrame, output_path: str | Path, digits: int = 4) -> pd.DataFrame:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    out = round_table(df, digits=digits)
    out.to_csv(output_path, index=False)

    return out


# ============================================================
# Table 6
# ============================================================

def make_table6_overall(test_metrics: pd.DataFrame) -> pd.DataFrame:
    """
    Table 6. Overall diagnostic decision-support performance.

    Grouping:
    - method

    Metrics:
    - Macro-F1
    - Evidence Precision
    - Evidence Recall
    - Unsupported Claim Rate
    - Unsafe Recommendation Rate
    - Generic Action Rate
    - Cost per Report
    """
    require_columns(
        test_metrics,
        [
            "method",
            "gold_label",
            "predicted_label",
            "evidence_precision",
            "evidence_recall",
            "unsupported_claim_rate",
            "unsafe_recommendation_rate",
            "generic_action_rate",
            "cost_per_report",
        ],
        "Table 6",
    )

    rows = []

    for method in ordered_methods(test_metrics["method"].unique()):
        group = test_metrics[test_metrics["method"] == method]
        metrics = aggregate_group_metrics(group, MAIN_METRICS)

        rows.append(
            {
                "Method": method,
                "Macro-F1": metrics["macro_f1"],
                "Evidence P": metrics["evidence_precision"],
                "Evidence R": metrics["evidence_recall"],
                "Unsupported": metrics["unsupported_claim_rate"],
                "Unsafe": metrics["unsafe_recommendation_rate"],
                "Generic Action": metrics["generic_action_rate"],
                "Cost/report": metrics["cost_per_report"],
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# Table 7
# ============================================================

def make_table7_datasetwise(test_metrics: pd.DataFrame) -> pd.DataFrame:
    """
    Table 7. Dataset-wise diagnostic decision-support performance.

    Grouping:
    - dataset
    - method

    Metrics are aligned with Table 6.
    """
    require_columns(
        test_metrics,
        [
            "dataset",
            "method",
            "gold_label",
            "predicted_label",
            "evidence_precision",
            "evidence_recall",
            "unsupported_claim_rate",
            "unsafe_recommendation_rate",
            "generic_action_rate",
            "cost_per_report",
        ],
        "Table 7",
    )

    rows = []

    for dataset in sorted(test_metrics["dataset"].dropna().unique()):
        dataset_group = test_metrics[test_metrics["dataset"] == dataset]

        for method in ordered_methods(dataset_group["method"].unique()):
            group = dataset_group[dataset_group["method"] == method]
            metrics = aggregate_group_metrics(group, DATASETWISE_METRICS)

            rows.append(
                {
                    "Dataset": dataset,
                    "Method": method,
                    "Macro-F1": metrics["macro_f1"],
                    "Evidence P": metrics["evidence_precision"],
                    "Evidence R": metrics["evidence_recall"],
                    "Unsupported": metrics["unsupported_claim_rate"],
                    "Unsafe": metrics["unsafe_recommendation_rate"],
                    "Generic Action": metrics["generic_action_rate"],
                    "Cost/report": metrics["cost_per_report"],
                }
            )

    return pd.DataFrame(rows)

# ============================================================
# Table 8
# ============================================================

def make_table8_component_effects(test_metrics: pd.DataFrame) -> pd.DataFrame:
    """
    Table 8. Dataset-wise component effects.

    This reports metric differences for each dataset:
    - M1 -> M2
    - M2 -> M3
    - M3 -> M4
    """
    require_columns(
        test_metrics,
        [
            "dataset",
            "method",
            "gold_label",
            "predicted_label",
            "evidence_recall",
            "source_hallucination_rate",
            "temporal_violation_rate",
            "evidence_linked_action_ratio",
            "unsafe_recommendation_rate",
            "causal_overclaim_rate",
        ],
        "Table 8",
    )

    comparisons = [
        ("M1 -> M2", "M1", "M2"),
        ("M2 -> M3", "M2", "M3"),
        ("M3 -> M4", "M3", "M4"),
    ]

    rows = []

    for dataset in sorted(test_metrics["dataset"].dropna().unique()):
        dataset_metrics = test_metrics[test_metrics["dataset"] == dataset]

        method_summary: dict[str, dict[str, float]] = {}

        for method in ordered_methods(dataset_metrics["method"].unique()):
            group = dataset_metrics[dataset_metrics["method"] == method]
            method_summary[method] = aggregate_group_metrics(group, COMPONENT_METRICS)

        for label, before, after in comparisons:
            row = {
                "Dataset": dataset,
                "Comparison": label,
            }

            for metric in COMPONENT_METRICS:
                if before in method_summary and after in method_summary:
                    before_val = method_summary[before].get(metric, np.nan)
                    after_val = method_summary[after].get(metric, np.nan)
                    row[metric] = after_val - before_val
                else:
                    row[metric] = np.nan

            rows.append(
                {
                    "Dataset": row["Dataset"],
                    "Comparison": row["Comparison"],
                    "Evidence R": row["evidence_recall"],
                    "Source Hallucination": row["source_hallucination_rate"],
                    "Temporal Violation": row["temporal_violation_rate"],
                    "Evidence-linked Action": row["evidence_linked_action_ratio"],
                    "Unsafe": row["unsafe_recommendation_rate"],
                    "Causal Overclaim": row["causal_overclaim_rate"],
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# Table 9
# ============================================================

def make_table9_optimization_trajectory(search_or_trajectory_metrics: pd.DataFrame) -> pd.DataFrame:
    """
    Table 9. Summary of domain-constrained optimization trajectory.

    Preferred input columns:
    - stage: one of {Best seed harness, Intermediate configured harness,
      Final screened candidate}
    - candidate_id
    - J_mfg
    - method or candidate_id
    - gold_label / predicted_label if macro_f1 must be computed
    - evidence_precision
    - unsupported_claim_rate
    - unsafe_recommendation_rate
    - cost_per_report

    If stage is absent, this function attempts to infer a compact trajectory:
    - Best seed harness: best row/group among candidates containing "seed"
    - Intermediate configured harness: median iteration if iteration column exists
    - Final screened candidate: candidate with highest J_mfg
    """
    df = search_or_trajectory_metrics.copy()

    if "stage" in df.columns:
        stage_order = [
            "Best seed harness",
            "Intermediate configured harness",
            "Final screened candidate",
        ]

        rows = []

        for stage in stage_order:
            group = df[df["stage"] == stage]
            if len(group) == 0:
                rows.append(_empty_table9_row(stage))
                continue

            rows.append(_summarize_optimization_stage(stage, group))

        return pd.DataFrame(rows)

    # Fallback: summarize all available rows as one final stage.
    return pd.DataFrame(
        [
            _summarize_optimization_stage(
                "Final screened candidate",
                df,
            )
        ]
    )


def _empty_table9_row(stage: str) -> dict[str, float | str]:
    return {
        "Candidate stage": stage,
        "J_mfg": np.nan,
        "Macro-F1": np.nan,
        "Evidence P": np.nan,
        "Unsupported": np.nan,
        "Unsafe": np.nan,
        "Cost/report": np.nan,
    }


def _summarize_optimization_stage(stage: str, group: pd.DataFrame) -> dict[str, float | str]:
    row: dict[str, float | str] = {"Candidate stage": stage}

    if "J_mfg" in group.columns:
        row["J_mfg"] = safe_mean(group["J_mfg"])
    else:
        row["J_mfg"] = np.nan

    if {"gold_label", "predicted_label"}.issubset(group.columns):
        row["Macro-F1"] = compute_macro_f1(group)
    elif "macro_f1" in group.columns:
        row["Macro-F1"] = safe_mean(group["macro_f1"])
    else:
        row["Macro-F1"] = np.nan

    row["Evidence P"] = safe_mean(group["evidence_precision"]) if "evidence_precision" in group.columns else np.nan
    row["Unsupported"] = safe_mean(group["unsupported_claim_rate"]) if "unsupported_claim_rate" in group.columns else np.nan
    row["Unsafe"] = safe_mean(group["unsafe_recommendation_rate"]) if "unsafe_recommendation_rate" in group.columns else np.nan
    row["Cost/report"] = safe_mean(group["cost_per_report"]) if "cost_per_report" in group.columns else np.nan

    return row


# ============================================================
# Table 10
# ============================================================

def make_table10_failure_trace_analysis(test_metrics: pd.DataFrame) -> pd.DataFrame:
    """
    Table 10. Dataset-wise failure trace analysis.

    This reports F1--F10 rates by dataset and method.
    """
    require_columns(test_metrics, ["dataset", "method", *FAILURE_FLAGS], "Table 10")

    rows = []

    for dataset in sorted(test_metrics["dataset"].dropna().unique()):
        dataset_group = test_metrics[test_metrics["dataset"] == dataset]

        for method in ordered_methods(dataset_group["method"].unique()):
            group = dataset_group[dataset_group["method"] == method]

            row = {
                "Dataset": dataset,
                "Method": method,
            }
            for flag in FAILURE_FLAGS:
                row[flag] = safe_mean(group[flag])

            rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# Table 11
# ============================================================

def make_table11_reliability_cost(test_metrics: pd.DataFrame) -> pd.DataFrame:
    """
    Table 11. Dataset-wise reliability--cost trade-off between M3 and M4.
    """
    require_columns(
        test_metrics,
        [
            "dataset",
            "method",
            "gold_label",
            "predicted_label",
            "evidence_precision",
            "unsupported_claim_rate",
            "unsafe_recommendation_rate",
            "cost_per_report",
        ],
        "Table 11",
    )

    rows = []

    for dataset in sorted(test_metrics["dataset"].dropna().unique()):
        dataset_group = test_metrics[test_metrics["dataset"] == dataset]

        for method in ["M3", "M4"]:
            group = dataset_group[dataset_group["method"] == method]

            if len(group) == 0:
                rows.append(
                    {
                        "Dataset": dataset,
                        "Method": method,
                        "Macro-F1": np.nan,
                        "Evidence P": np.nan,
                        "Unsupported": np.nan,
                        "Unsafe": np.nan,
                        "Cost/report": np.nan,
                    }
                )
                continue

            metrics = aggregate_group_metrics(group, RELIABILITY_COST_METRICS)

            rows.append(
                {
                    "Dataset": dataset,
                    "Method": method,
                    "Macro-F1": metrics["macro_f1"],
                    "Evidence P": metrics["evidence_precision"],
                    "Unsupported": metrics["unsupported_claim_rate"],
                    "Unsafe": metrics["unsafe_recommendation_rate"],
                    "Cost/report": metrics["cost_per_report"],
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# All-table runner
# ============================================================

def make_coverage_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    """Show empty reports and the effective denominators beside macro rates.

    Legacy inputs without counts cannot support this table. Return an explicit
    empty table rather than inventing coverage from a zero violation rate.
    """
    needed={'n_generated_evidence','n_required_proxy_evidence','n_generated_actions',
            'evidence_precision','evidence_recall','evidence_linked_action_ratio','unsafe_recommendation_rate'}
    if not needed.issubset(metrics.columns):
        return pd.DataFrame(columns=['dataset','method','reports','coverage_unavailable'])
    rows=[]
    for (dataset,method),g in metrics.groupby(['dataset','method'],sort=False):
        ne=g.n_generated_evidence;na=g.n_generated_actions
        active=g.loc[na>0]
        rows.append({'dataset':dataset,'method':method,'reports':len(g),
            'empty_evidence_reports':int((ne==0).sum()),'evidence_coverage':float((ne>0).mean()),
            'recall_defined_reports':int((g.n_required_proxy_evidence>0).sum()),
            'evidence_precision_macro':safe_mean(g.evidence_precision),
            'evidence_recall_macro_defined':safe_mean(g.evidence_recall),
            'empty_action_reports':int((na==0).sum()),'action_coverage':float((na>0).mean()),
            'evidence_linked_action_macro_defined':safe_mean(g.evidence_linked_action_ratio),
            'unsafe_macro_all_reports':safe_mean(g.unsafe_recommendation_rate),
            'unsafe_macro_nonempty_actions':safe_mean(active.unsafe_recommendation_rate),
            'generated_actions':int(na.sum()),
            'evidence_linked_action_coverage':float((g.n_evidence_linked_actions>0).mean()) if 'n_evidence_linked_actions' in g else float('nan')})
    return pd.DataFrame(rows)


def aggregate_all_tables(
    *,
    test_metrics_path: str | Path,
    output_dir: str | Path,
    search_metrics_path: str | Path | None = None,
    validation_metrics_path: str | Path | None = None,
    digits: int = 4,
) -> dict[str, pd.DataFrame]:
    """
    Generate Table 6--11 CSV files.

    Parameters
    ----------
    test_metrics_path:
        Path to final test task-level metrics.
    output_dir:
        Directory where table CSV files will be saved.
    search_metrics_path:
        Optional search/trajectory metrics for Table 9.
    validation_metrics_path:
        Optional validation metrics. Currently not required but retained for
        future selection diagnostics.
    digits:
        Number of decimals for saved tables.
    """
    output_dir = Path(output_dir)
    ensure_dir(output_dir)

    test_metrics = read_metrics_csv(test_metrics_path)

    tables: dict[str, pd.DataFrame] = {}

    tables["table6_overall"] = make_table6_overall(test_metrics)
    tables["table7_datasetwise"] = make_table7_datasetwise(test_metrics)
    tables["table8_component_effects"] = make_table8_component_effects(test_metrics)
    tables["table10_failure_trace_analysis"] = make_table10_failure_trace_analysis(test_metrics)
    tables["table11_reliability_cost"] = make_table11_reliability_cost(test_metrics)
    tables["coverage_summary"] = make_coverage_summary(test_metrics)

    if search_metrics_path is not None and Path(search_metrics_path).exists():
        search_metrics = read_metrics_csv(search_metrics_path)
        tables["table9_optimization_trajectory"] = make_table9_optimization_trajectory(search_metrics)
    else:
        tables["table9_optimization_trajectory"] = pd.DataFrame(
            [
                _empty_table9_row("Best seed harness"),
                _empty_table9_row("Intermediate configured harness"),
                _empty_table9_row("Final screened candidate"),
            ]
        )

    filename_map = {
        "table6_overall": "table6_overall.csv",
        "table7_datasetwise": "table7_datasetwise.csv",
        "table8_component_effects": "table8_component_effects.csv",
        "table9_optimization_trajectory": "table9_optimization_trajectory.csv",
        "table10_failure_trace_analysis": "table10_failure_trace_analysis.csv",
        "table11_reliability_cost": "table11_reliability_cost.csv",
        "coverage_summary": "coverage_summary.csv",
    }

    for table_name, df in tables.items():
        write_table(df, output_dir / filename_map[table_name], digits=digits)

    return tables


# ============================================================
# CLI
# ============================================================

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Aggregate Table 6--11 outputs.")
    parser.add_argument(
        "--test-metrics",
        required=True,
        help="Path to final test metrics_by_task.csv",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where table CSV files will be written.",
    )
    parser.add_argument(
        "--search-metrics",
        default=None,
        help="Optional path to search/trajectory metrics CSV for Table 9.",
    )
    parser.add_argument(
        "--validation-metrics",
        default=None,
        help="Optional path to validation metrics CSV.",
    )
    parser.add_argument(
        "--digits",
        type=int,
        default=4,
        help="Number of decimal places in saved tables.",
    )

    args = parser.parse_args()

    tables = aggregate_all_tables(
        test_metrics_path=args.test_metrics,
        output_dir=args.output_dir,
        search_metrics_path=args.search_metrics,
        validation_metrics_path=args.validation_metrics,
        digits=args.digits,
    )

    print("Generated tables:")
    for table_name, df in tables.items():
        print(f"- {table_name}: {df.shape}")


if __name__ == "__main__":
    main()
