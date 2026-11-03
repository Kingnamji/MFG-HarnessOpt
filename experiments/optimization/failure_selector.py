"""
Failure trace selector for harness optimization.

This module selects representative failure traces from TraceRepository outputs.

Input:
    outputs/search/<run_name>/
    ├─ metrics_by_task.csv
    ├─ failure_traces.csv
    ├─ reports.jsonl
    └─ traces.jsonl

Primary use:
    Step 20. Select compact, diverse, high-value failure bundles that can be
    passed to future harness_proposer.py.

Design principles:
- Prioritize manufacturing-relevant failures:
  F10 repair failure, F8 high-risk action violation, F7 action without evidence,
  F6 causal overclaim, F5 source hallucination, F2 evidence omission, F1 label error.
- Keep dataset/method diversity.
- Avoid selecting only repeated variants of the same task.
- Export proposer-ready JSONL and CSV summaries.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from experiments.optimization.trace_repository import (
    DEFAULT_FAILURE_PRIORITY,
    FAILURE_CATEGORIES,
    FAILURE_FLAGS,
    TraceBundle,
    TraceRepository,
    load_trace_repository,
    normalize_failure_id,
    normalize_method,
)


# ============================================================
# Selection config
# ============================================================

@dataclass(frozen=True)
class FailureSelectionConfig:
    max_bundles: int = 20

    # Main priority order. Earlier = more important.
    priority_failure_ids: list[str] = field(default_factory=lambda: DEFAULT_FAILURE_PRIORITY.copy())

    # Optional hard filter. If None, all failure types are eligible.
    include_failure_ids: list[str] | None = None

    # Optional method/dataset filters.
    methods: list[str] | None = None
    datasets: list[str] | None = None

    # Diversity controls.
    max_per_failure_id: int | None = None
    max_per_dataset: int | None = None
    max_per_method: int | None = None
    max_per_task: int = 1
    task_ids: list[str] | None = None
    min_label_error_bundles: int = 0
    min_verified_bundles: int = 0

    # Selection behavior.
    require_failure: bool = True
    include_report: bool = True
    prefer_m3_failures: bool = True
    prefer_repair_failures: bool = True
    prefer_unsafe_or_unsupported: bool = True

    # Lower cost traces can be useful, but failure severity is more important.
    cost_penalty_weight: float = 0.10


@dataclass(frozen=True)
class SelectedFailureTrace:
    rank: int
    score: float
    bundle: TraceBundle
    selection_reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        row = self.bundle.to_dict()
        row["rank"] = self.rank
        row["selection_score"] = self.score
        row["selection_reasons"] = self.selection_reasons
        return row


# ============================================================
# Scoring
# ============================================================

def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
    except Exception:
        pass

    try:
        return float(value)
    except Exception:
        return default


def _safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}

    return bool(value)


def normalize_failure_ids(failure_ids: Iterable[str] | None) -> list[str] | None:
    if failure_ids is None:
        return None
    return [normalize_failure_id(f) for f in failure_ids]


def score_failure_bundle(
    bundle: TraceBundle,
    config: FailureSelectionConfig,
) -> tuple[float, list[str]]:
    """
    Score a failure bundle.

    Higher score means the bundle is more useful for harness optimization.
    """
    reasons: list[str] = []
    score = 0.0

    priority = [normalize_failure_id(f) for f in config.priority_failure_ids]
    priority_weights = {
        failure_id: len(priority) - idx
        for idx, failure_id in enumerate(priority)
    }

    for failure_id in bundle.failure_ids:
        weight = priority_weights.get(failure_id, 1)
        score += float(weight)
        reasons.append(f"{failure_id}:{FAILURE_CATEGORIES.get(failure_id, 'Unknown')}")

    vc = bundle.violation_context

    unsupported = _safe_float(vc.get("unsupported_claim_rate"))
    unsafe = _safe_float(vc.get("unsafe_recommendation_rate"))
    generic = _safe_float(vc.get("generic_action_rate"))
    evidence_recall = _safe_float(vc.get("evidence_recall"), default=1.0)
    evidence_precision = _safe_float(vc.get("evidence_precision"), default=1.0)
    cost = _safe_float(vc.get("cost_per_report"))
    repair_count = _safe_float(vc.get("repair_count"))
    repair_failed = _safe_bool(vc.get("repair_failed"))

    if config.prefer_unsafe_or_unsupported:
        if unsafe > 0:
            score += 3.0 * unsafe
            reasons.append(f"unsafe={unsafe:.4f}")
        if unsupported > 0:
            score += 2.0 * unsupported
            reasons.append(f"unsupported={unsupported:.4f}")
        if generic > 0:
            score += 1.0 * generic
            reasons.append(f"generic={generic:.4f}")

    if evidence_recall < 1.0:
        score += 1.5 * (1.0 - evidence_recall)
        reasons.append(f"evidence_recall_gap={1.0 - evidence_recall:.4f}")

    if evidence_precision < 1.0:
        score += 1.0 * (1.0 - evidence_precision)
        reasons.append(f"evidence_precision_gap={1.0 - evidence_precision:.4f}")

    if config.prefer_repair_failures:
        if repair_failed:
            score += 5.0
            reasons.append("repair_failed=True")
        elif repair_count > 0:
            score += 1.0
            reasons.append(f"repair_count={repair_count:.0f}")

    if config.prefer_m3_failures and normalize_method(bundle.method) in {"M3","M4"}:
        score += 1.0
        reasons.append(f"method={normalize_method(bundle.method)}")

    # Cost is penalized lightly. We still want high-severity failures.
    if cost > 0:
        score -= config.cost_penalty_weight * cost
        reasons.append(f"cost={cost:.6f}")

    return score, reasons


# ============================================================
# Selection
# ============================================================

def _bundle_primary_failure(bundle: TraceBundle, priority_failure_ids: list[str]) -> str:
    bundle_failures = set(bundle.failure_ids)
    for failure_id in priority_failure_ids:
        if failure_id in bundle_failures:
            return failure_id
    if bundle.failure_ids:
        return bundle.failure_ids[0]
    return "NONE"


def _passes_quota(
    bundle: TraceBundle,
    *,
    config: FailureSelectionConfig,
    counters: dict[str, Counter],
    primary_failure: str,
) -> bool:
    dataset = bundle.dataset
    method = normalize_method(bundle.method)
    task_id = bundle.task_id

    if config.max_per_task is not None:
        if counters["task"][task_id] >= config.max_per_task:
            return False

    if config.max_per_dataset is not None:
        if counters["dataset"][dataset] >= config.max_per_dataset:
            return False

    if config.max_per_method is not None:
        if counters["method"][method] >= config.max_per_method:
            return False

    if config.max_per_failure_id is not None and primary_failure != "NONE":
        if counters["failure"][primary_failure] >= config.max_per_failure_id:
            return False

    return True


def _update_counters(
    bundle: TraceBundle,
    *,
    counters: dict[str, Counter],
    primary_failure: str,
) -> None:
    counters["task"][bundle.task_id] += 1
    counters["dataset"][bundle.dataset] += 1
    counters["method"][normalize_method(bundle.method)] += 1

    if primary_failure != "NONE":
        counters["failure"][primary_failure] += 1


def select_failure_traces(
    repo: TraceRepository,
    config: FailureSelectionConfig,
) -> list[SelectedFailureTrace]:
    """
    Select representative failure traces from a repository.
    """
    include_failure_ids = normalize_failure_ids(config.include_failure_ids)
    priority_failure_ids = normalize_failure_ids(config.priority_failure_ids) or DEFAULT_FAILURE_PRIORITY

    bundles = repo.build_failure_bundles(
        methods=config.methods,
        datasets=config.datasets,
        task_ids=config.task_ids,
        failure_ids=include_failure_ids,
        max_bundles=None,
        include_report=config.include_report,
        prioritize=False,
    )

    if config.require_failure:
        bundles = [b for b in bundles if len(b.failure_ids) > 0]

    scored: list[tuple[float, list[str], TraceBundle, str]] = []

    for bundle in bundles:
        score, reasons = score_failure_bundle(bundle, config)
        primary_failure = _bundle_primary_failure(bundle, priority_failure_ids)
        scored.append((score, reasons, bundle, primary_failure))

    scored.sort(
        key=lambda item: (
            -item[0],
            item[2].dataset,
            normalize_method(item[2].method),
            item[2].task_id,
        )
    )

    selected: list[SelectedFailureTrace] = []
    counters: dict[str, Counter] = {
        "task": Counter(),
        "dataset": Counter(),
        "method": Counter(),
        "failure": Counter(),
    }
    identity=lambda b:(b.task_id,normalize_method(b.method),b.harness_id)
    selected_keys=set()
    def add(item):
        score,reasons,bundle,primary_failure=item
        if identity(bundle) in selected_keys or len(selected)>=config.max_bundles:return False
        if not _passes_quota(bundle,config=config,counters=counters,primary_failure=primary_failure):return False
        _update_counters(bundle,counters=counters,primary_failure=primary_failure)
        selected_keys.add(identity(bundle))
        selected.append(SelectedFailureTrace(rank=len(selected)+1,score=score,bundle=bundle,selection_reasons=reasons))
        return True
    for predicate,minimum in [
        (lambda b:'F1' in b.failure_ids,config.min_label_error_bundles),
        (lambda b:normalize_method(b.method) in {'M3','M4'},config.min_verified_bundles)]:
        for item in scored:
            if sum(predicate(s.bundle) for s in selected)>=minimum:break
            if predicate(item[2]):add(item)

    # First pass: respect all quotas.
    for score, reasons, bundle, primary_failure in scored:
        if len(selected) >= config.max_bundles:
            break

        if identity(bundle) in selected_keys:continue
        if not _passes_quota(
            bundle,
            config=config,
            counters=counters,
            primary_failure=primary_failure,
        ):
            continue

        _update_counters(bundle, counters=counters, primary_failure=primary_failure)
        selected_keys.add(identity(bundle))
        selected.append(
            SelectedFailureTrace(
                rank=len(selected) + 1,
                score=score,
                bundle=bundle,
                selection_reasons=reasons,
            )
        )

    # Second pass: if quotas were too strict, fill remaining slots while still
    # respecting max_per_task to avoid duplicates of the same task.
    if len(selected) < config.max_bundles:
        selected_keys = {
            identity(s.bundle)
            for s in selected
        }

        for score, reasons, bundle, primary_failure in scored:
            if len(selected) >= config.max_bundles:
                break

            key = identity(bundle)
            if key in selected_keys:
                continue

            if config.max_per_task is not None:
                if counters["task"][bundle.task_id] >= config.max_per_task:
                    continue

            _update_counters(bundle, counters=counters, primary_failure=primary_failure)
            selected_keys.add(key)
            selected.append(
                SelectedFailureTrace(
                    rank=len(selected) + 1,
                    score=score,
                    bundle=bundle,
                    selection_reasons=reasons + ["selected_in_fill_pass"],
                )
            )

    return selected


# ============================================================
# Output helpers
# ============================================================

def selected_to_dataframe(selected: list[SelectedFailureTrace]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for item in selected:
        bundle = item.bundle
        vc = bundle.violation_context

        rows.append(
            {
                "rank": item.rank,
                "selection_score": item.score,
                "task_id": bundle.task_id,
                "dataset": bundle.dataset,
                "split": bundle.split,
                "method": bundle.method,
                "harness_id": bundle.harness_id,
                "failure_ids": ",".join(bundle.failure_ids),
                "failure_categories": "; ".join(bundle.failure_categories),
                "gold_label": bundle.gold_label,
                "predicted_label": bundle.predicted_label,
                "label_correct": vc.get("label_correct"),
                "evidence_precision": vc.get("evidence_precision"),
                "evidence_recall": vc.get("evidence_recall"),
                "unsupported_claim_rate": vc.get("unsupported_claim_rate"),
                "unsafe_recommendation_rate": vc.get("unsafe_recommendation_rate"),
                "generic_action_rate": vc.get("generic_action_rate"),
                "repair_count": vc.get("repair_count"),
                "repair_failed": vc.get("repair_failed"),
                "cost_per_report": vc.get("cost_per_report"),
                "selection_reasons": " | ".join(item.selection_reasons),
            }
        )

    return pd.DataFrame(rows)


def write_selected_jsonl(
    selected: list[SelectedFailureTrace],
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for item in selected:
            f.write(json.dumps(item.to_dict(), ensure_ascii=False, default=str) + "\n")


def write_selected_csv(
    selected: list[SelectedFailureTrace],
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = selected_to_dataframe(selected)
    df.to_csv(output_path, index=False)


def make_selection_summary(selected: list[SelectedFailureTrace]) -> dict[str, Any]:
    failure_counter: Counter = Counter()
    dataset_counter: Counter = Counter()
    method_counter: Counter = Counter()

    for item in selected:
        dataset_counter[item.bundle.dataset] += 1
        method_counter[normalize_method(item.bundle.method)] += 1
        for failure_id in item.bundle.failure_ids:
            failure_counter[failure_id] += 1

    return {
        "n_selected": len(selected),
        "datasets": dict(dataset_counter),
        "methods": dict(method_counter),
        "failure_ids": dict(failure_counter),
        "top_tasks": [
            {
                "rank": item.rank,
                "task_id": item.bundle.task_id,
                "dataset": item.bundle.dataset,
                "method": item.bundle.method,
                "score": item.score,
                "failure_ids": item.bundle.failure_ids,
            }
            for item in selected[:5]
        ],
    }


def write_summary_json(summary: dict[str, Any], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)


# ============================================================
# Config construction
# ============================================================

def make_config_from_args(args: argparse.Namespace) -> FailureSelectionConfig:
    include_failure_ids = None
    if args.failure_ids:
        include_failure_ids = [
            normalize_failure_id(f)
            for f in args.failure_ids.split(",")
            if f.strip()
        ]

    methods = None
    if args.methods:
        methods = [
            normalize_method(m)
            for m in args.methods.split(",")
            if m.strip()
        ]

    datasets = None
    if args.datasets:
        datasets = [
            d.strip()
            for d in args.datasets.split(",")
            if d.strip()
        ]

    priority_failure_ids = DEFAULT_FAILURE_PRIORITY.copy()
    if args.priority_failure_ids:
        priority_failure_ids = [
            normalize_failure_id(f)
            for f in args.priority_failure_ids.split(",")
            if f.strip()
        ]

    return FailureSelectionConfig(
        max_bundles=args.max_bundles,
        priority_failure_ids=priority_failure_ids,
        include_failure_ids=include_failure_ids,
        methods=methods,
        datasets=datasets,
        max_per_failure_id=args.max_per_failure_id,
        max_per_dataset=args.max_per_dataset,
        max_per_method=args.max_per_method,
        max_per_task=args.max_per_task,
        include_report=not args.exclude_report,
        prefer_m3_failures=not args.no_prefer_m3,
        prefer_repair_failures=not args.no_prefer_repair_failures,
        prefer_unsafe_or_unsupported=not args.no_prefer_unsafe_or_unsupported,
        cost_penalty_weight=args.cost_penalty_weight,
    )


# ============================================================
# CLI
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select representative failure traces for harness optimization."
    )

    parser.add_argument(
        "--root-dir",
        required=True,
        help="Search output directory, e.g. outputs/search/api_small_test",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Default: <root-dir>/selected_failures",
    )
    parser.add_argument(
        "--max-bundles",
        type=int,
        default=20,
        help="Maximum number of selected failure bundles.",
    )
    parser.add_argument(
        "--failure-ids",
        default=None,
        help="Optional comma-separated failure IDs to include, e.g. F2,F7,F10.",
    )
    parser.add_argument(
        "--priority-failure-ids",
        default=None,
        help="Optional comma-separated failure priority order.",
    )
    parser.add_argument(
        "--methods",
        default=None,
        help="Optional comma-separated methods, e.g. M2,M3.",
    )
    parser.add_argument(
        "--datasets",
        default=None,
        help="Optional comma-separated dataset names, e.g. 'AI4I 2020,Azure PdM'.",
    )
    parser.add_argument(
        "--max-per-failure-id",
        type=int,
        default=None,
        help="Optional max selected bundles per primary failure ID.",
    )
    parser.add_argument(
        "--max-per-dataset",
        type=int,
        default=None,
        help="Optional max selected bundles per dataset.",
    )
    parser.add_argument(
        "--max-per-method",
        type=int,
        default=None,
        help="Optional max selected bundles per method.",
    )
    parser.add_argument(
        "--max-per-task",
        type=int,
        default=1,
        help="Max selected bundles per task. Default: 1.",
    )
    parser.add_argument(
        "--exclude-report",
        action="store_true",
        help="Do not include structured report in selected JSONL output.",
    )
    parser.add_argument(
        "--no-prefer-m3",
        action="store_true",
        help="Disable M3 bonus in scoring.",
    )
    parser.add_argument(
        "--no-prefer-repair-failures",
        action="store_true",
        help="Disable repair failure bonus in scoring.",
    )
    parser.add_argument(
        "--no-prefer-unsafe-or-unsupported",
        action="store_true",
        help="Disable continuous unsafe/unsupported severity bonus.",
    )
    parser.add_argument(
        "--cost-penalty-weight",
        type=float,
        default=0.10,
        help="Small cost penalty weight. Default: 0.10.",
    )

    args = parser.parse_args()

    root_dir = Path(args.root_dir)
    output_dir = Path(args.output_dir) if args.output_dir else root_dir / "selected_failures"
    output_dir.mkdir(parents=True, exist_ok=True)

    repo = load_trace_repository(root_dir)
    config = make_config_from_args(args)

    selected = select_failure_traces(repo, config)
    selected_df = selected_to_dataframe(selected)
    summary = make_selection_summary(selected)

    jsonl_path = output_dir / "selected_failure_bundles.jsonl"
    csv_path = output_dir / "selected_failure_bundles.csv"
    summary_path = output_dir / "selection_summary.json"

    write_selected_jsonl(selected, jsonl_path)
    write_selected_csv(selected, csv_path)
    write_summary_json(summary, summary_path)

    print("=== Failure Selection Summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))

    print("\n=== Selected Failure Bundles ===")
    if selected_df.empty:
        print("No failure bundles selected.")
    else:
        preview_cols = [
            "rank",
            "selection_score",
            "task_id",
            "dataset",
            "method",
            "failure_ids",
            "gold_label",
            "predicted_label",
            "unsupported_claim_rate",
            "unsafe_recommendation_rate",
            "repair_failed",
        ]
        print(selected_df[[col for col in preview_cols if col in selected_df.columns]].head(20))

    print("\nSaved outputs:")
    print(f"- {jsonl_path}")
    print(f"- {csv_path}")
    print(f"- {summary_path}")


if __name__ == "__main__":
    main()
