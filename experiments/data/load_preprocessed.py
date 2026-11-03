"""
Load preprocessed datasets for manufacturing diagnostic harness experiments.

This module loads:
- AI4I task table, feature table, proxy evidence, allowed schema, metric config
- Azure PdM task table, telemetry summary, event windows, proxy evidence,
  allowed schema, metric config, action ontology, action preconditions,
  generic action rules

Expected project structure:

Code/
├─ Dataset/
│  └─ Preprocessed/
│     ├─ AI4I/
│     │  ├─ tasks_ai4i.csv
│     │  ├─ ai4i_features.csv
│     │  ├─ proxy_evidence_ai4i.csv
│     │  ├─ allowed_schema_ai4i.json
│     │  └─ metric_config_ai4i.json
│     └─ PdM/
│        ├─ tasks_azure.csv
│        ├─ azure_telemetry_summary.csv
│        ├─ event_windows_azure.csv
│        ├─ proxy_evidence_azure.csv
│        ├─ allowed_schema_azure.json
│        ├─ metric_config_azure.json
│        ├─ action_ontology.json
│        ├─ action_preconditions.json
│        └─ generic_action_rules.json
└─ experiments/
   └─ data/
      └─ load_preprocessed.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from .generation_evidence import validate_generation_evidence, validate_observation_windows


DatasetName = Literal["AI4I", "PdM", "Azure", "Azure PdM"]


# ============================================================
# Path utilities
# ============================================================

def get_project_root() -> Path:
    """
    Return the project root.

    This function assumes it is called from within:
        Code/experiments/data/load_preprocessed.py

    It resolves:
        load_preprocessed.py -> data -> experiments -> Code
    """
    return Path(__file__).resolve().parents[2]


def get_dataset_dir(project_root: Path | None = None) -> Path:
    root = project_root or get_project_root()
    return root / "Dataset"


def get_preprocessed_dir(project_root: Path | None = None) -> Path:
    return get_dataset_dir(project_root) / "Preprocessed"


def get_ai4i_dir(project_root: Path | None = None) -> Path:
    return get_preprocessed_dir(project_root) / "AI4I"


def get_pdm_dir(project_root: Path | None = None) -> Path:
    return get_preprocessed_dir(project_root) / "PdM"


def normalize_dataset_name(dataset: str) -> str:
    normalized = dataset.strip().lower()

    if normalized in {"ai4i", "ai4i 2020"}:
        return "AI4I"

    if normalized in {"pdm", "azure", "azure pdm", "microsoft azure predictive maintenance"}:
        return "PdM"

    raise ValueError(
        f"Unsupported dataset name: {dataset}. "
        "Use one of: 'AI4I', 'PdM', 'Azure', 'Azure PdM'."
    )


# ============================================================
# I/O utilities
# ============================================================

def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return path


def load_json(path: Path) -> dict[str, Any]:
    require_file(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_csv(path: Path, parse_dates: list[str] | None = None) -> pd.DataFrame:
    require_file(path)
    return pd.read_csv(path, parse_dates=parse_dates)


# ============================================================
# Data containers
# ============================================================

@dataclass(frozen=True)
class AI4IPreprocessedData:
    tasks: pd.DataFrame
    features: pd.DataFrame
    proxy_evidence: pd.DataFrame
    allowed_schema: dict[str, Any]
    metric_config: dict[str, Any]
    generation_evidence: pd.DataFrame


@dataclass(frozen=True)
class PdMPreprocessedData:
    tasks: pd.DataFrame
    telemetry_summary: pd.DataFrame
    event_windows: pd.DataFrame
    proxy_evidence: pd.DataFrame
    allowed_schema: dict[str, Any]
    metric_config: dict[str, Any]
    action_ontology: dict[str, Any]
    action_preconditions: dict[str, Any]
    generic_action_rules: dict[str, Any]
    association_guide: dict[str, Any]
    generation_evidence: pd.DataFrame


@dataclass(frozen=True)
class AllPreprocessedData:
    ai4i: AI4IPreprocessedData
    pdm: PdMPreprocessedData


# ============================================================
# Load AI4I
# ============================================================

def load_ai4i_preprocessed(
    project_root: Path | None = None,
) -> AI4IPreprocessedData:
    ai4i_dir = get_ai4i_dir(project_root)

    tasks = load_csv(ai4i_dir / "tasks_ai4i.csv")
    features = load_csv(ai4i_dir / "ai4i_features.csv")
    proxy_evidence = load_csv(ai4i_dir / "proxy_evidence_ai4i.csv")
    generation_evidence = load_csv(ai4i_dir / "generation_evidence_ai4i.csv")
    validate_generation_evidence(generation_evidence, tasks.task_id)
    validate_observation_windows(generation_evidence, tasks)

    allowed_schema = load_json(ai4i_dir / "allowed_schema_ai4i.json")
    metric_config = load_json(ai4i_dir / "metric_config_ai4i.json")

    return AI4IPreprocessedData(
        tasks=tasks,
        features=features,
        proxy_evidence=proxy_evidence,
        allowed_schema=allowed_schema,
        metric_config=metric_config,
        generation_evidence=generation_evidence,
    )


# ============================================================
# Load Azure PdM
# ============================================================

AZURE_ASSOCIATION_GUIDE_PATH = (
    Path.cwd()
    / "Dataset"
    / "Preprocessed"
    / "PdM"
    / "azure_search_association_guide.json"
)

def load_json_if_exists(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        return {}

    return data


def load_azure_association_guide(
    path: str | Path,
) -> dict[str, Any]:
    """
    Load search-split-only Azure component association guide.

    Returns an empty dict if the guide has not been generated yet.
    """
    return load_json_if_exists(path)
    

def load_pdm_preprocessed(
    project_root: Path | None = None,
) -> PdMPreprocessedData:
    pdm_dir = get_pdm_dir(project_root)

    tasks = load_csv(pdm_dir / "tasks_azure.csv", parse_dates=["failure_time"])
    from experiments.data.azure_observation_context import attach_contexts
    tasks = attach_contexts(tasks, pdm_dir)
    telemetry_summary = load_csv(pdm_dir / "azure_telemetry_summary.csv")
    event_windows = load_csv(
        pdm_dir / "event_windows_azure.csv",
        parse_dates=["failure_time", "start_time", "end_time"],
    )
    proxy_evidence = load_csv(
        pdm_dir / "proxy_evidence_azure.csv",
        parse_dates=["event_time"],
    )

    allowed_schema = load_json(pdm_dir / "allowed_schema_azure.json")
    metric_config = load_json(pdm_dir / "metric_config_azure.json")
    action_ontology = load_json(pdm_dir / "action_ontology.json")
    action_preconditions = load_json(pdm_dir / "action_preconditions.json")
    generic_action_rules = load_json(pdm_dir / "generic_action_rules.json")

    association_guide = load_azure_association_guide(pdm_dir / "azure_search_association_guide.json")
    generation_evidence = load_csv(pdm_dir / "generation_evidence_azure.csv", parse_dates=["event_time"])
    validate_generation_evidence(generation_evidence, tasks.task_id)
    validate_observation_windows(generation_evidence, tasks)

    return PdMPreprocessedData(
        tasks=tasks,
        telemetry_summary=telemetry_summary,
        event_windows=event_windows,
        proxy_evidence=proxy_evidence,
        allowed_schema=allowed_schema,
        metric_config=metric_config,
        action_ontology=action_ontology,
        action_preconditions=action_preconditions,
        generic_action_rules=generic_action_rules,
        association_guide=association_guide,
        generation_evidence=generation_evidence,
    )


def load_all_preprocessed(
    project_root: Path | None = None,
) -> AllPreprocessedData:
    return AllPreprocessedData(
        ai4i=load_ai4i_preprocessed(project_root),
        pdm=load_pdm_preprocessed(project_root),
    )


# ============================================================
# Task/evidence access helpers
# ============================================================

def get_tasks_by_split(
    tasks: pd.DataFrame,
    split: Literal["search", "validation", "test"] | str,
) -> pd.DataFrame:
    if "split" not in tasks.columns:
        raise KeyError("tasks DataFrame must contain a 'split' column.")

    return tasks[tasks["split"] == split].copy().reset_index(drop=True)


def get_proxy_evidence_for_task(
    proxy_evidence: pd.DataFrame,
    task_id: str,
) -> pd.DataFrame:
    if "task_id" not in proxy_evidence.columns:
        raise KeyError("proxy_evidence DataFrame must contain a 'task_id' column.")

    return proxy_evidence[proxy_evidence["task_id"] == task_id].copy().reset_index(drop=True)


def get_task_by_id(
    tasks: pd.DataFrame,
    task_id: str,
) -> dict[str, Any]:
    if "task_id" not in tasks.columns:
        raise KeyError("tasks DataFrame must contain a 'task_id' column.")

    matched = tasks[tasks["task_id"] == task_id]

    if len(matched) == 0:
        raise KeyError(f"task_id not found: {task_id}")

    if len(matched) > 1:
        raise ValueError(f"task_id is not unique: {task_id}")

    return matched.iloc[0].to_dict()


def get_task_and_evidence(
    tasks: pd.DataFrame,
    proxy_evidence: pd.DataFrame,
    task_id: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    task = get_task_by_id(tasks, task_id)
    evidence = get_proxy_evidence_for_task(proxy_evidence, task_id)
    return task, evidence


def get_dataset_bundle(
    dataset: DatasetName,
    project_root: Path | None = None,
) -> AI4IPreprocessedData | PdMPreprocessedData:
    name = normalize_dataset_name(dataset)

    if name == "AI4I":
        return load_ai4i_preprocessed(project_root)

    if name == "PdM":
        return load_pdm_preprocessed(project_root)

    raise RuntimeError("Unreachable dataset branch.")


# ============================================================
# Validation and summary
# ============================================================

def validate_ai4i_preprocessed(data: AI4IPreprocessedData) -> dict[str, Any]:
    required_task_cols = {
        "task_id",
        "dataset",
        "split",
        "gold_label",
    }

    required_evidence_cols = {
        "task_id",
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

    missing_task_cols = required_task_cols - set(data.tasks.columns)
    missing_evidence_cols = required_evidence_cols - set(data.proxy_evidence.columns)

    split_counts = data.tasks["split"].value_counts().to_dict()
    label_counts = data.tasks["gold_label"].value_counts().to_dict()

    return {
        "valid": len(missing_task_cols) == 0 and len(missing_evidence_cols) == 0,
        "tasks_shape": data.tasks.shape,
        "features_shape": data.features.shape,
        "proxy_evidence_shape": data.proxy_evidence.shape,
        "missing_task_columns": sorted(missing_task_cols),
        "missing_evidence_columns": sorted(missing_evidence_cols),
        "split_counts": split_counts,
        "label_counts": label_counts,
        "n_tasks_with_evidence": int(data.proxy_evidence["task_id"].nunique()),
    }


def validate_pdm_preprocessed(data: PdMPreprocessedData) -> dict[str, Any]:
    required_task_cols = {
        "task_id",
        "dataset",
        "split",
        "machineID",
        "failure_time",
        "gold_label",
    }

    required_evidence_cols = {
        "task_id",
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

    missing_task_cols = required_task_cols - set(data.tasks.columns)
    missing_evidence_cols = required_evidence_cols - set(data.proxy_evidence.columns)

    split_counts = data.tasks["split"].value_counts().to_dict()
    label_counts = data.tasks["gold_label"].value_counts().to_dict()

    machine_leakage_count = 0
    if "machineID" in data.tasks.columns and "split" in data.tasks.columns:
        split_per_machine = data.tasks.groupby("machineID")["split"].nunique()
        machine_leakage_count = int((split_per_machine > 1).sum())

    return {
        "valid": (
            len(missing_task_cols) == 0
            and len(missing_evidence_cols) == 0
            and machine_leakage_count == 0
        ),
        "tasks_shape": data.tasks.shape,
        "telemetry_summary_shape": data.telemetry_summary.shape,
        "event_windows_shape": data.event_windows.shape,
        "proxy_evidence_shape": data.proxy_evidence.shape,
        "missing_task_columns": sorted(missing_task_cols),
        "missing_evidence_columns": sorted(missing_evidence_cols),
        "split_counts": split_counts,
        "label_counts": label_counts,
        "n_tasks_with_evidence": int(data.proxy_evidence["task_id"].nunique()),
        "machine_leakage_count": machine_leakage_count,
        "evidence_source_counts": data.proxy_evidence["source"].value_counts().to_dict(),
        "evidence_role_counts": data.proxy_evidence["role"].value_counts().to_dict(),
        "association_guide_loaded": bool(data.association_guide),
        "association_guide_top_associations": len(
            data.association_guide.get("top_associations", [])
        ) if data.association_guide else 0,
        "association_guide_search_tasks": (
            data.association_guide.get("search_split_summary", {}).get("n_tasks")
            if data.association_guide
            else None
        ),
    }


def print_preprocessed_summary(project_root: Path | None = None) -> None:
    ai4i = load_ai4i_preprocessed(project_root)
    pdm = load_pdm_preprocessed(project_root)

    ai4i_summary = validate_ai4i_preprocessed(ai4i)
    pdm_summary = validate_pdm_preprocessed(pdm)

    print("=== AI4I Preprocessed Summary ===")
    for key, value in ai4i_summary.items():
        print(f"{key}: {value}")

    print("\n=== Azure PdM Preprocessed Summary ===")
    for key, value in pdm_summary.items():
        print(f"{key}: {value}")


# ============================================================
# Script entry point
# ============================================================

if __name__ == "__main__":
    print_preprocessed_summary()
