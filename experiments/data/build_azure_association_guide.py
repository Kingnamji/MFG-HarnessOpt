"""
Build search-split-only Azure PdM component association guide.

Purpose
-------
This script builds an aggregate, leakage-controlled association guide for
Azure PdM component-failure diagnosis.

It uses only:
- Azure search split tasks
- search split gold labels
- search split proxy evidence

It does NOT expose:
- task-level gold labels to the LLM
- task-level supports values to the LLM
- validation/test labels
- validation/test evidence-label associations

Outputs
-------
Dataset/Preprocessed/PdM/azure_search_association_guide.json
Dataset/Preprocessed/PdM/azure_search_association_guide.txt

Recommended use
---------------
The prompt can use this as a weak prior for Azure PdM only:

- errorID -> component association
- telemetry field/direction -> component association
- maintenance comp value -> component association

The guide should be described as search-derived aggregate statistics, not
ground truth for the current task.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path.cwd() / "Dataset" / "Preprocessed" / "PdM"

TASKS_PATH = BASE_DIR / "tasks_azure.csv"
EVIDENCE_PATH = BASE_DIR / "generation_evidence_azure.csv"

GUIDE_JSON_OUT = BASE_DIR / "azure_search_association_guide.json"
GUIDE_TXT_OUT = BASE_DIR / "azure_search_association_guide.txt"


# ============================================================
# Constants
# ============================================================

CANDIDATE_LABELS = ["comp1", "comp2", "comp3", "comp4"]

DEFAULT_MIN_COUNT = 5
DEFAULT_TOP_K_PER_GROUP = 8


# ============================================================
# I/O
# ============================================================

def read_csv_required(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return pd.read_csv(path)


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


def write_text(path: str | Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(text)


# ============================================================
# Helpers
# ============================================================

def normalize_component_distribution(counts: pd.Series) -> dict[str, float]:
    total = float(counts.sum())
    if total <= 0:
        return {label: 0.0 for label in CANDIDATE_LABELS}

    return {
        label: round(float(counts.get(label, 0.0)) / total, 4)
        for label in CANDIDATE_LABELS
    }


def entropy_from_distribution(dist: dict[str, float]) -> float:
    import math

    entropy = 0.0
    for p in dist.values():
        if p > 0:
            entropy -= p * math.log(p, 2)
    return round(entropy, 4)


def confidence_from_distribution(dist: dict[str, float]) -> str:
    values = sorted(dist.values(), reverse=True)
    if not values:
        return "low"

    top = values[0]
    second = values[1] if len(values) > 1 else 0.0
    margin = top - second

    if top >= 0.60 and margin >= 0.20:
        return "high"
    if top >= 0.45 and margin >= 0.10:
        return "medium"
    return "low"


def make_association_record(
    *,
    association_type: str,
    evidence_key: str,
    source: str,
    field: str,
    value: str | None,
    direction: str | None,
    group: pd.DataFrame,
) -> dict[str, Any]:
    label_counts = group["gold_label"].value_counts()
    dist = normalize_component_distribution(label_counts)

    top_component = max(dist, key=dist.get)
    top_probability = dist[top_component]

    return {
        "type": association_type,
        "evidence_key": evidence_key,
        "source": source,
        "field": field,
        "value": value,
        "direction": direction,
        "n": int(len(group)),
        "component_counts": {
            label: int(label_counts.get(label, 0))
            for label in CANDIDATE_LABELS
        },
        "component_distribution": dist,
        "top_component": top_component,
        "top_probability": round(float(top_probability), 4),
        "entropy": entropy_from_distribution(dist),
        "confidence": confidence_from_distribution(dist),
    }


def filter_and_rank_records(
    records: list[dict[str, Any]],
    *,
    min_count: int,
    top_k: int,
) -> list[dict[str, Any]]:
    filtered = [r for r in records if int(r.get("n", 0)) >= min_count]

    filtered = sorted(
        filtered,
        key=lambda r: (
            r.get("top_probability", 0.0),
            r.get("n", 0),
            -r.get("entropy", 99.0),
        ),
        reverse=True,
    )

    return filtered[:top_k]


# ============================================================
# Association builders
# ============================================================

def build_error_associations(df: pd.DataFrame, *, min_count: int, top_k: int) -> list[dict[str, Any]]:
    sub = df[
        (df["source"] == "errors")
        & (df["field"] == "errorID")
        & df["value"].notna()
    ].copy()

    records: list[dict[str, Any]] = []

    for value, group in sub.groupby("value"):
        records.append(
            make_association_record(
                association_type="error_component_association",
                evidence_key=f"errors:errorID:{value}",
                source="errors",
                field="errorID",
                value=str(value),
                direction=None,
                group=group,
            )
        )

    return filter_and_rank_records(records, min_count=min_count, top_k=top_k)


def build_telemetry_associations(df: pd.DataFrame, *, min_count: int, top_k: int) -> list[dict[str, Any]]:
    sub = df[
        (df["source"] == "telemetry")
        & df["field"].notna()
        & df["direction"].notna()
    ].copy()

    # Do not use unavailable/normal as diagnostic priors.
    sub = sub[~sub["direction"].astype(str).isin(["unavailable", "normal"])]

    records: list[dict[str, Any]] = []

    for (field, direction), group in sub.groupby(["field", "direction"]):
        records.append(
            make_association_record(
                association_type="telemetry_component_association",
                evidence_key=f"telemetry:{field}:{direction}",
                source="telemetry",
                field=str(field),
                value=None,
                direction=str(direction),
                group=group,
            )
        )

    return filter_and_rank_records(records, min_count=min_count, top_k=top_k)


def build_maintenance_associations(df: pd.DataFrame, *, min_count: int, top_k: int) -> list[dict[str, Any]]:
    sub = df[
        (df["source"] == "maintenance")
        & (df["field"] == "comp")
        & df["value"].notna()
    ].copy()

    records: list[dict[str, Any]] = []

    for value, group in sub.groupby("value"):
        records.append(
            make_association_record(
                association_type="maintenance_component_association",
                evidence_key=f"maintenance:comp:{value}",
                source="maintenance",
                field="comp",
                value=str(value),
                direction=None,
                group=group,
            )
        )

    ranked = filter_and_rank_records(records, min_count=min_count, top_k=top_k)

    for record in ranked:
        record["note"] = "Maintenance evidence is contextual by default and should not be treated as causal by itself."

    return ranked


# ============================================================
# Guide construction
# ============================================================

def build_azure_association_guide(
    *,
    tasks_path: str | Path = TASKS_PATH,
    evidence_path: str | Path = EVIDENCE_PATH,
    min_count: int = DEFAULT_MIN_COUNT,
    top_k_per_group: int = DEFAULT_TOP_K_PER_GROUP,
) -> dict[str, Any]:
    tasks = read_csv_required(tasks_path)
    evidence = read_csv_required(evidence_path)

    required_task_cols = {"task_id", "split", "gold_label"}
    missing_task_cols = required_task_cols - set(tasks.columns)
    if missing_task_cols:
        raise ValueError(f"tasks file missing columns: {sorted(missing_task_cols)}")

    required_evidence_cols = {"task_id", "source", "field", "value", "direction"}
    missing_evidence_cols = required_evidence_cols - set(evidence.columns)
    if missing_evidence_cols:
        raise ValueError(f"evidence file missing columns: {sorted(missing_evidence_cols)}")

    search_tasks = tasks[tasks["split"] == "search"].copy()

    if search_tasks.empty:
        raise ValueError("No Azure search split tasks found.")

    invalid_labels = sorted(set(search_tasks["gold_label"].astype(str)) - set(CANDIDATE_LABELS))
    if invalid_labels:
        raise ValueError(f"Unexpected Azure labels in search split: {invalid_labels}")

    # Search split only.
    search_evidence = evidence[evidence["task_id"].isin(search_tasks["task_id"])].copy()

    # Join gold label only for aggregate association computation.
    # This joined table must never be given to the LLM task-by-task.
    joined = search_evidence.merge(
        search_tasks[["task_id", "gold_label"]],
        on="task_id",
        how="inner",
    )

    error_associations = build_error_associations(
        joined,
        min_count=min_count,
        top_k=top_k_per_group,
    )

    telemetry_associations = build_telemetry_associations(
        joined,
        min_count=min_count,
        top_k=top_k_per_group,
    )

    maintenance_associations = build_maintenance_associations(
        joined,
        min_count=min_count,
        top_k=top_k_per_group,
    )

    top_associations = (
        error_associations
        + telemetry_associations
        + maintenance_associations
    )

    top_associations = sorted(
        top_associations,
        key=lambda r: (
            r.get("confidence") == "high",
            r.get("top_probability", 0.0),
            r.get("n", 0),
        ),
        reverse=True,
    )

    guide = {
        "dataset": "Azure PdM",
        "guide_name": "azure_search_association_guide",
        "description": (
            "Aggregate component association guide computed only from the Azure PdM search split. "
            "Use as weak prior; do not treat as task-level ground truth."
        ),
        "leakage_control": {
            "uses_split": "search",
            "uses_validation_labels": False,
            "uses_test_labels": False,
            "exposes_task_level_gold_label": False,
            "exposes_task_level_supports": False,
            "allowed_use": "weak_prior_for_prompting",
        },
        "configuration": {
            "min_count": min_count,
            "top_k_per_group": top_k_per_group,
            "candidate_labels": CANDIDATE_LABELS,
        },
        "search_split_summary": {
            "n_tasks": int(len(search_tasks)),
            "n_tasks_with_evidence": int(search_evidence["task_id"].nunique()),
            "label_counts": {
                label: int((search_tasks["gold_label"] == label).sum())
                for label in CANDIDATE_LABELS
            },
            "evidence_source_counts": search_evidence["source"].value_counts().to_dict()
            if "source" in search_evidence.columns
            else {},
        },
        "error_associations": error_associations,
        "telemetry_associations": telemetry_associations,
        "maintenance_associations": maintenance_associations,
        "top_associations": top_associations,
    }

    return guide


def render_guide_text(guide: dict[str, Any], *, max_lines: int = 30) -> str:
    lines = [
        "Search-derived Azure component association guide",
        "================================================",
        "",
        "Leakage control:",
        "- Computed only from the Azure PdM search split.",
        "- Does not expose validation/test labels.",
        "- Does not expose task-level hidden supports.",
        "- Use as weak prior, not as ground truth for the current task.",
        "",
        "Search split summary:",
    ]

    summary = guide.get("search_split_summary", {})
    lines.append(f"- n_tasks: {summary.get('n_tasks')}")
    lines.append(f"- n_tasks_with_evidence: {summary.get('n_tasks_with_evidence')}")
    lines.append(f"- label_counts: {summary.get('label_counts')}")
    lines.append(f"- evidence_source_counts: {summary.get('evidence_source_counts')}")
    lines.append("")
    lines.append("Top associations:")

    top = guide.get("top_associations", [])[:max_lines]

    for item in top:
        key = item.get("evidence_key")
        n = item.get("n")
        top_component = item.get("top_component")
        prob = item.get("top_probability")
        confidence = item.get("confidence")
        dist = item.get("component_distribution")
        note = item.get("note")

        line = (
            f"- {key}: top={top_component}, p={prob}, "
            f"n={n}, confidence={confidence}, distribution={dist}"
        )

        if note:
            line += f" ({note})"

        lines.append(line)

    return "\n".join(lines) + "\n"


# ============================================================
# CLI
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build search-split-only Azure PdM component association guide."
    )

    parser.add_argument(
        "--tasks-path",
        default=str(TASKS_PATH),
        help="Path to tasks_azure.csv.",
    )
    parser.add_argument(
        "--evidence-path",
        default=str(EVIDENCE_PATH),
        help="Path to proxy_evidence_azure.csv.",
    )
    parser.add_argument(
        "--output-json",
        default=str(GUIDE_JSON_OUT),
        help="Output JSON path.",
    )
    parser.add_argument(
        "--output-txt",
        default=str(GUIDE_TXT_OUT),
        help="Output text path.",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=DEFAULT_MIN_COUNT,
        help="Minimum support count for an association to be included.",
    )
    parser.add_argument(
        "--top-k-per-group",
        type=int,
        default=DEFAULT_TOP_K_PER_GROUP,
        help="Maximum associations per source group.",
    )

    args = parser.parse_args()

    guide = build_azure_association_guide(
        tasks_path=args.tasks_path,
        evidence_path=args.evidence_path,
        min_count=args.min_count,
        top_k_per_group=args.top_k_per_group,
    )

    guide_text = render_guide_text(guide)

    write_json(args.output_json, guide)
    write_text(args.output_txt, guide_text)

    print("=== Azure Association Guide Summary ===")
    print(f"tasks_path: {args.tasks_path}")
    print(f"evidence_path: {args.evidence_path}")
    print(f"output_json: {args.output_json}")
    print(f"output_txt: {args.output_txt}")
    print(f"min_count: {args.min_count}")
    print(f"top_k_per_group: {args.top_k_per_group}")
    print()

    print("Search split summary:")
    print(json.dumps(guide["search_split_summary"], indent=2, ensure_ascii=False))
    print()

    print("Association counts:")
    print(f"- error_associations: {len(guide['error_associations'])}")
    print(f"- telemetry_associations: {len(guide['telemetry_associations'])}")
    print(f"- maintenance_associations: {len(guide['maintenance_associations'])}")
    print(f"- top_associations: {len(guide['top_associations'])}")
    print()

    print("Top association preview:")
    for item in guide["top_associations"][:10]:
        print(
            f"- {item['evidence_key']} -> {item['top_component']} "
            f"(p={item['top_probability']}, n={item['n']}, confidence={item['confidence']})"
        )


if __name__ == "__main__":
    main()
