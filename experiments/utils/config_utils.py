"""
Configuration utilities for manufacturing diagnostic harness experiments.

This module centralizes YAML loading, path resolution, and experiment config
composition. Runners should use this module instead of hard-coding paths,
model settings, method settings, and optimization weights.

Expected structure:

experiments/
├─ configs/
│  ├─ paths.yaml
│  ├─ model_config.yaml
│  ├─ experiment_config.yaml
│  ├─ metric_weights.yaml
│  └─ methods/
│     ├─ M0_zero_shot.yaml
│     ├─ M1_static_summary.yaml
│     ├─ M2_tool_source.yaml
│     └─ M3_verified.yaml
└─ utils/
   └─ config_utils.py
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


# ============================================================
# Constants
# ============================================================

METHOD_CONFIG_FILES = {
    "M0": "M0_zero_shot.yaml",
    "M1": "M1_static_summary.yaml",
    "M2": "M2_tool_source.yaml",
    "M3": "M3_verified.yaml",
}


# ============================================================
# Data containers
# ============================================================

@dataclass(frozen=True)
class ExperimentConfigs:
    project_root: Path
    paths: dict[str, Any]
    model: dict[str, Any]
    experiment: dict[str, Any]
    metric_weights: dict[str, Any]
    methods: dict[str, dict[str, Any]]

    def get_method_config(self, method_id: str) -> dict[str, Any]:
        method_id = method_id.upper()
        if method_id not in self.methods:
            raise KeyError(
                f"Unknown method config: {method_id}. "
                f"Available: {sorted(self.methods.keys())}"
            )
        return self.methods[method_id]

    def get_model_profile(self, profile_name: str | None = None) -> dict[str, Any]:
        if profile_name is None:
            profile_name = self.model.get("default_model_profile")

        profiles = self.model.get("profiles", {})
        if profile_name not in profiles:
            raise KeyError(
                f"Unknown model profile: {profile_name}. "
                f"Available: {sorted(profiles.keys())}"
            )

        profile = copy.deepcopy(profiles[profile_name])
        profile["profile_name"] = profile_name
        return profile


# ============================================================
# Path helpers
# ============================================================

def get_project_root() -> Path:
    """
    Resolve project root from this file location.

    config_utils.py -> utils -> experiments -> project root
    """
    return Path(__file__).resolve().parents[2]


def get_configs_dir(project_root: Path | None = None) -> Path:
    root = project_root or get_project_root()
    return root / "experiments" / "configs"


def resolve_path(path_value: str | Path, project_root: Path | None = None) -> Path:
    """
    Resolve a relative path against project root.
    Absolute paths are returned unchanged.
    """
    root = project_root or get_project_root()
    path = Path(path_value)

    if path.is_absolute():
        return path

    return root / path


def resolve_nested_paths(
    obj: Any,
    project_root: Path | None = None,
    key_hint: str | None = None,
) -> Any:
    """
    Recursively resolve path-like values in a nested YAML object.

    This function resolves values only for keys that look path-related:
    - keys ending in _dir, _path, _json, _csv
    - keys in a small known set such as tasks, features, proxy_evidence, etc.

    It leaves regular strings such as model names untouched.
    """
    path_like_keys = {
        "root_dir",
        "preprocessed_dir",
        "dir",
        "tasks",
        "features",
        "proxy_evidence",
        "allowed_schema",
        "metric_config",
        "telemetry_summary",
        "event_windows",
        "action_ontology",
        "action_preconditions",
        "generic_action_rules",
        "openai_api_key_json",
        "smoke_test_dir",
        "smoke_test_api_dir",
        "search_dir",
        "validation_dir",
        "final_test_dir",
        "tables_dir",
        "logs_dir",
        "traces_dir",
        "reports_dir",
        "docs_dir",
    }

    if isinstance(obj, dict):
        return {
            key: resolve_nested_paths(value, project_root, key_hint=str(key))
            for key, value in obj.items()
        }

    if isinstance(obj, list):
        return [
            resolve_nested_paths(value, project_root, key_hint=key_hint)
            for value in obj
        ]

    if isinstance(obj, str):
        key = key_hint or ""
        if (
            key in path_like_keys
            or key.endswith("_dir")
            or key.endswith("_path")
            or key.endswith("_json")
            or key.endswith("_csv")
        ):
            return resolve_path(obj, project_root)

    return obj


# ============================================================
# YAML I/O
# ============================================================

def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        return {}

    if not isinstance(data, dict):
        raise TypeError(f"YAML file must contain a mapping/dict: {path}")

    return data


def save_yaml(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    serializable = convert_paths_to_strings(data)

    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            serializable,
            f,
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
        )


def convert_paths_to_strings(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, dict):
        return {
            key: convert_paths_to_strings(value)
            for key, value in obj.items()
        }

    if isinstance(obj, list):
        return [
            convert_paths_to_strings(value)
            for value in obj
        ]

    return obj


# ============================================================
# Config loading
# ============================================================

def load_paths_config(
    project_root: Path | None = None,
    *,
    resolve_paths: bool = True,
) -> dict[str, Any]:
    root = project_root or get_project_root()
    config = load_yaml(get_configs_dir(root) / "paths.yaml")

    if resolve_paths:
        config = resolve_nested_paths(config, root)

    return config


def load_model_config(project_root: Path | None = None) -> dict[str, Any]:
    root = project_root or get_project_root()
    return load_yaml(get_configs_dir(root) / "model_config.yaml")


def load_experiment_config(project_root: Path | None = None) -> dict[str, Any]:
    root = project_root or get_project_root()
    return load_yaml(get_configs_dir(root) / "experiment_config.yaml")


def load_metric_weights_config(project_root: Path | None = None) -> dict[str, Any]:
    root = project_root or get_project_root()
    return load_yaml(get_configs_dir(root) / "metric_weights.yaml")


def load_method_config(
    method_id: str,
    project_root: Path | None = None,
) -> dict[str, Any]:
    method_id = method_id.upper()
    root = project_root or get_project_root()

    if method_id not in METHOD_CONFIG_FILES:
        raise KeyError(
            f"Unknown method ID: {method_id}. "
            f"Available: {sorted(METHOD_CONFIG_FILES.keys())}"
        )

    path = get_configs_dir(root) / "methods" / METHOD_CONFIG_FILES[method_id]
    config = load_yaml(path)

    yaml_method_id = str(config.get("method_id", "")).upper()
    if yaml_method_id and yaml_method_id != method_id:
        raise ValueError(
            f"Method config mismatch. Expected {method_id}, "
            f"but YAML contains method_id={yaml_method_id}: {path}"
        )

    return config


def load_all_method_configs(
    project_root: Path | None = None,
    methods: list[str] | None = None,
) -> dict[str, dict[str, Any]]:
    methods = methods or list(METHOD_CONFIG_FILES.keys())

    return {
        method.upper(): load_method_config(method, project_root)
        for method in methods
    }


def load_experiment_configs(
    project_root: Path | None = None,
    *,
    resolve_paths: bool = True,
) -> ExperimentConfigs:
    root = project_root or get_project_root()

    paths = load_paths_config(root, resolve_paths=resolve_paths)
    model = load_model_config(root)
    experiment = load_experiment_config(root)
    metric_weights = load_metric_weights_config(root)
    methods = load_all_method_configs(root)

    return ExperimentConfigs(
        project_root=root,
        paths=paths,
        model=model,
        experiment=experiment,
        metric_weights=metric_weights,
        methods=methods,
    )


# ============================================================
# Config access helpers
# ============================================================

def get_default_model_profile(configs: ExperimentConfigs) -> dict[str, Any]:
    return configs.get_model_profile(configs.model.get("default_model_profile"))


def get_diagnosis_model_profile(configs: ExperimentConfigs) -> dict[str, Any]:
    return configs.get_model_profile(configs.model.get("diagnosis_model_profile"))


def get_proposer_model_profile(configs: ExperimentConfigs) -> dict[str, Any]:
    return configs.get_model_profile(configs.model.get("proposer_model_profile"))


def get_smoke_model_profile(configs: ExperimentConfigs) -> dict[str, Any]:
    return configs.get_model_profile(configs.model.get("smoke_model_profile"))


def get_api_key_path(configs: ExperimentConfigs) -> Path:
    return configs.paths["secrets"]["openai_api_key_json"]


def get_output_dir(configs: ExperimentConfigs, key: str) -> Path:
    outputs = configs.paths.get("outputs", {})
    if key not in outputs:
        raise KeyError(
            f"Unknown output dir key: {key}. "
            f"Available: {sorted(outputs.keys())}"
        )

    path = outputs[key]
    if not isinstance(path, Path):
        path = resolve_path(path, configs.project_root)

    return path


def get_split_sizes(
    configs: ExperimentConfigs,
    split: str,
) -> dict[str, int]:
    splits = configs.experiment.get("splits", {})
    if split not in splits:
        raise KeyError(
            f"Unknown split: {split}. Available: {sorted(splits.keys())}"
        )

    return copy.deepcopy(splits[split])


def get_methods_for_stage(
    configs: ExperimentConfigs,
    stage: str,
) -> list[str]:
    """
    Get method list from experiment_config.yaml.

    Supported examples:
    - smoke_test
    - pilot
    - baselines
    - final_test
    """
    if stage in configs.experiment:
        section = configs.experiment[stage]
        if isinstance(section, dict) and "methods" in section:
            return [str(m).upper() for m in section["methods"]]

    methods_section = configs.experiment.get("methods", {})
    if stage in methods_section:
        return [str(m).upper() for m in methods_section[stage]]

    raise KeyError(
        f"Could not find methods for stage={stage}. "
        "Check experiment_config.yaml."
    )


def validate_no_supports_exposure(configs: ExperimentConfigs) -> None:
    """
    Guardrail: ensure M2/M3 do not expose hidden 'supports' annotations
    in LLM input.
    """
    for method_id in ["M2", "M3"]:
        method_config = configs.get_method_config(method_id)
        llm_input = method_config.get("llm_input", {})

        if bool(llm_input.get("expose_supports", False)):
            raise ValueError(
                f"{method_id} config violates anti-leakage rule: "
                "expose_supports must be false."
            )

        evidence_cols = llm_input.get("evidence_table_columns", [])
        if "supports" in evidence_cols:
            raise ValueError(
                f"{method_id} config violates anti-leakage rule: "
                "'supports' must not appear in evidence_table_columns."
            )


def validate_configs(configs: ExperimentConfigs) -> dict[str, Any]:
    """
    Lightweight consistency checks. Returns a summary dict.
    """
    errors: list[str] = []
    warnings: list[str] = []

    try:
        validate_no_supports_exposure(configs)
    except Exception as exc:
        errors.append(str(exc))

    # Check model profiles.
    for profile_key in [
        "default_model_profile",
        "diagnosis_model_profile",
        "proposer_model_profile",
        "smoke_model_profile",
    ]:
        profile_name = configs.model.get(profile_key)
        profiles = configs.model.get("profiles", {})
        if profile_name not in profiles:
            errors.append(f"Missing model profile for {profile_key}: {profile_name}")

    # Check method configs.
    for method_id, method_config in configs.methods.items():
        if method_config.get("method_id", "").upper() != method_id:
            errors.append(f"Method ID mismatch for {method_id}")

    # Check key path existence as warning, not hard error.
    api_key_path = get_api_key_path(configs)
    if not api_key_path.exists():
        warnings.append(f"API key JSON does not exist yet: {api_key_path}")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "project_root": str(configs.project_root),
        "loaded_methods": sorted(configs.methods.keys()),
        "default_model_profile": configs.model.get("default_model_profile"),
        "diagnosis_model_profile": configs.model.get("diagnosis_model_profile"),
    }


# ============================================================
# CLI smoke check
# ============================================================

def main() -> None:
    configs = load_experiment_configs()
    summary = validate_configs(configs)

    print("=== Experiment Config Summary ===")
    for key, value in summary.items():
        print(f"{key}: {value}")

    print("\n=== Key Paths ===")
    print(f"project_root: {configs.project_root}")
    print(f"api_key_path: {get_api_key_path(configs)}")
    print(f"search_dir: {get_output_dir(configs, 'search_dir')}")
    print(f"final_test_dir: {get_output_dir(configs, 'final_test_dir')}")

    print("\n=== Model Profiles ===")
    print(f"default: {get_default_model_profile(configs)}")
    print(f"diagnosis: {get_diagnosis_model_profile(configs)}")
    print(f"proposer: {get_proposer_model_profile(configs)}")

    print("\n=== Method Configs ===")
    for method_id, method_config in configs.methods.items():
        print(
            f"{method_id}: "
            f"name={method_config.get('name')}, "
            f"expose_supports={method_config.get('llm_input', {}).get('expose_supports')}"
        )


if __name__ == "__main__":
    main()