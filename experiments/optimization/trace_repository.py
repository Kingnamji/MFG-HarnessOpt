"""
Trace repository for harness optimization.

This module loads and queries search-run outputs produced by:

    experiments/runners/run_search.py

Expected input directory:

    outputs/search/<run_name>/
    ├─ reports.jsonl
    ├─ traces.jsonl
    ├─ metrics_by_task.csv
    ├─ failure_traces.csv
    ├─ search_summary.csv
    └─ config_snapshot.json

Primary responsibilities:
- Load reports, execution traces, task-level metrics, and failure traces.
- Provide method/dataset/failure-type filtering utilities.
- Build compact failure bundles for future failure_selector.py and harness_proposer.py.
- Keep raw traces accessible without coupling optimization logic to file formats.

This module does not call the LLM and does not modify harness code.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


# ============================================================
# Constants
# ============================================================

FAILURE_FLAGS = [f"F{i}" for i in range(1, 11)]

FAILURE_CATEGORIES = {
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

DEFAULT_FAILURE_PRIORITY = [
    "F10",
    "F8",
    "F7",
    "F6",
    "F5",
    "F2",
    "F1",
    "F9",
    "F4",
    "F3",
]


# ============================================================
# I/O helpers
# ============================================================

def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required trace repository file not found: {path}")
    return path


def read_json(path: Path) -> dict[str, Any]:
    require_file(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    require_file(path)

    rows: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc

    return rows


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:return pd.read_csv(path)
    except pd.errors.EmptyDataError:return pd.DataFrame()


def normalize_method(method: str) -> str:
    return str(method).strip().upper()


def initial_report_from_trace(trace):
    """Extract the first generated report only, never expose raw API payloads."""
    raw=(trace or {}).get('raw_response') or {}
    if (isinstance(raw, dict) and isinstance(raw.get('initial_report'), dict)
            and (raw.get('repair_report') is not None or (raw.get('action_filter') or {}).get('interventions'))):
        return raw['initial_report']
    while isinstance(raw,dict) and 'responses' in raw:raw=raw['responses']
    first=raw.get('first_response') if isinstance(raw,dict) else None
    if not isinstance(first,dict):return None
    try:
        text=first.get('output_text') or ''.join(block.get('text','') for item in first.get('output',[])
            for block in item.get('content',[]))
        report=json.loads(text)
        return report if isinstance(report,dict) else None
    except (ValueError,TypeError,AttributeError):return None


def normalize_failure_id(failure_id: str) -> str:
    value = str(failure_id).strip().upper()
    if not value.startswith("F"):
        value = f"F{value}"
    return value


# ============================================================
# Data classes
# ============================================================

@dataclass(frozen=True)
class TraceRepositoryPaths:
    root_dir: Path
    reports_jsonl: Path
    traces_jsonl: Path
    metrics_csv: Path
    failure_traces_csv: Path
    search_summary_csv: Path
    config_snapshot_json: Path

    @classmethod
    def from_root(cls, root_dir: str | Path) -> "TraceRepositoryPaths":
        root = Path(root_dir)

        return cls(
            root_dir=root,
            reports_jsonl=root / "reports.jsonl",
            traces_jsonl=root / "traces.jsonl",
            metrics_csv=root / "metrics_by_task.csv",
            failure_traces_csv=root / "failure_traces.csv",
            search_summary_csv=root / "search_summary.csv",
            config_snapshot_json=root / "config_snapshot.json",
        )


@dataclass(frozen=True)
class TraceBundle:
    """
    Compact trace bundle passed to failure selectors or harness proposers.
    """

    task_id: str
    dataset: str
    split: str
    method: str
    harness_id: str

    failure_ids: list[str]
    failure_categories: list[str]

    gold_label: str | None
    predicted_label: str | None

    report: dict[str, Any] | None
    metric_row: dict[str, Any]
    execution_log: dict[str, Any]
    violation_context: dict[str, Any]
    raw_response_available: bool
    M3_reference: dict[str, Any] | None = None
    initial_report: dict[str, Any] | None = None
    action_filter_audit: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "dataset": self.dataset,
            "split": self.split,
            "method": self.method,
            "harness_id": self.harness_id,
            "failure_ids": self.failure_ids,
            "failure_categories": self.failure_categories,
            "gold_label": self.gold_label,
            "predicted_label": self.predicted_label,
            "report": self.report,
            "metric_row": self.metric_row,
            "execution_log": self.execution_log,
            "violation_context": self.violation_context,
            "raw_response_available": self.raw_response_available,
            "M3_reference": self.M3_reference,
            "initial_report": self.initial_report,
            "action_filter_audit": self.action_filter_audit,
        }


# ============================================================
# Repository
# ============================================================

class TraceRepository:
    """
    Query interface over one search-run output directory.
    """

    def __init__(
        self,
        root_dir: str | Path,
        *,
        load_raw_traces: bool = True,
    ) -> None:
        self.paths = TraceRepositoryPaths.from_root(root_dir)
        self.root_dir = self.paths.root_dir

        self.config_snapshot = self._load_config_snapshot()
        self.metrics = self._load_metrics()
        self.failure_traces = self._load_failure_traces()
        self.search_summary = self._load_search_summary()

        self.reports = self._load_reports()
        self.traces = self._load_traces() if load_raw_traces else []

        self._report_index = self._build_report_index(self.reports)
        self._trace_index = self._build_trace_index(self.traces)

    # --------------------------------------------------------
    # Loading
    # --------------------------------------------------------

    def _load_config_snapshot(self) -> dict[str, Any]:
        if self.paths.config_snapshot_json.exists():
            return read_json(self.paths.config_snapshot_json)
        return {}

    def _load_metrics(self) -> pd.DataFrame:
        require_file(self.paths.metrics_csv)
        df = pd.read_csv(self.paths.metrics_csv)

        if "method" in df.columns:
            df["method"] = df["method"].astype(str).str.upper()

        return df

    def _load_failure_traces(self) -> pd.DataFrame:
        df = read_csv_if_exists(self.paths.failure_traces_csv)

        if df.empty:
            return df

        if "method" in df.columns:
            df["method"] = df["method"].astype(str).str.upper()

        if "failure_id" in df.columns:
            df["failure_id"] = df["failure_id"].astype(str).str.upper()

        return df

    def _load_search_summary(self) -> pd.DataFrame:
        df = read_csv_if_exists(self.paths.search_summary_csv)

        if not df.empty and "method" in df.columns:
            df["method"] = df["method"].astype(str).str.upper()

        return df

    def _load_reports(self) -> list[dict[str, Any]]:
        if self.paths.reports_jsonl.exists():
            return read_jsonl(self.paths.reports_jsonl)
        return []

    def _load_traces(self) -> list[dict[str, Any]]:
        if self.paths.traces_jsonl.exists():
            return read_jsonl(self.paths.traces_jsonl)
        return []

    @staticmethod
    def _make_key(row: dict[str, Any]) -> tuple[str, str]:
        task_id = str(row.get("task_id", ""))
        method = normalize_method(row.get("method", ""))
        return task_id, method

    def _build_report_index(
        self,
        rows: list[dict[str, Any]],
    ) -> dict[tuple[str, str], dict[str, Any]]:
        return {
            self._make_key(row): row
            for row in rows
        }

    def _build_trace_index(
        self,
        rows: list[dict[str, Any]],
    ) -> dict[tuple[str, str], dict[str, Any]]:
        return {
            self._make_key(row): row
            for row in rows
        }

    # --------------------------------------------------------
    # Basic metadata
    # --------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        return {
            "root_dir": str(self.root_dir),
            "n_metrics": int(len(self.metrics)),
            "n_failure_traces": int(len(self.failure_traces)),
            "n_reports": int(len(self.reports)),
            "n_traces": int(len(self.traces)),
            "datasets": sorted(self.metrics["dataset"].dropna().unique().tolist())
            if "dataset" in self.metrics.columns else [],
            "methods": sorted(self.metrics["method"].dropna().unique().tolist())
            if "method" in self.metrics.columns else [],
            "failure_counts": self.failure_counts().to_dict()
            if not self.failure_traces.empty else {},
        }

    def failure_counts(self) -> pd.Series:
        if self.failure_traces.empty or "failure_id" not in self.failure_traces.columns:
            return pd.Series(dtype=int)
        return self.failure_traces["failure_id"].value_counts().sort_index()

    def method_counts(self) -> pd.Series:
        if "method" not in self.metrics.columns:
            return pd.Series(dtype=int)
        return self.metrics["method"].value_counts().sort_index()

    # --------------------------------------------------------
    # Filtering
    # --------------------------------------------------------

    def filter_metrics(
        self,
        *,
        methods: Iterable[str] | None = None,
        datasets: Iterable[str] | None = None,
        task_ids: Iterable[str] | None = None,
        only_failed: bool = False,
        failure_ids: Iterable[str] | None = None,
    ) -> pd.DataFrame:
        df = self.metrics.copy()

        if methods is not None:
            method_set = {normalize_method(m) for m in methods}
            df = df[df["method"].isin(method_set)]

        if datasets is not None:
            dataset_set = {str(d) for d in datasets}
            df = df[df["dataset"].isin(dataset_set)]

        if task_ids is not None:
            task_id_set = {str(t) for t in task_ids}
            df = df[df["task_id"].astype(str).isin(task_id_set)]

        if only_failed:
            available_flags = [flag for flag in FAILURE_FLAGS if flag in df.columns]
            if available_flags:
                flag_sum = df[available_flags].fillna(0).astype(int).sum(axis=1)
                df = df[flag_sum > 0]

        if failure_ids is not None:
            failure_ids_norm = [normalize_failure_id(f) for f in failure_ids]
            available_flags = [flag for flag in failure_ids_norm if flag in df.columns]
            if available_flags:
                flag_sum = df[available_flags].fillna(0).astype(int).sum(axis=1)
                df = df[flag_sum > 0]
            else:
                df = df.iloc[0:0]

        return df.reset_index(drop=True)

    def filter_failure_traces(
        self,
        *,
        methods: Iterable[str] | None = None,
        datasets: Iterable[str] | None = None,
        failure_ids: Iterable[str] | None = None,
        task_ids: Iterable[str] | None = None,
    ) -> pd.DataFrame:
        df = self.failure_traces.copy()

        if df.empty:
            return df

        if methods is not None:
            method_set = {normalize_method(m) for m in methods}
            df = df[df["method"].isin(method_set)]

        if datasets is not None:
            dataset_set = {str(d) for d in datasets}
            df = df[df["dataset"].isin(dataset_set)]

        if failure_ids is not None:
            failure_set = {normalize_failure_id(f) for f in failure_ids}
            df = df[df["failure_id"].isin(failure_set)]

        if task_ids is not None:
            task_id_set = {str(t) for t in task_ids}
            df = df[df["task_id"].astype(str).isin(task_id_set)]

        return df.reset_index(drop=True)

    # --------------------------------------------------------
    # Lookup
    # --------------------------------------------------------

    def get_metric_row(self, task_id: str, method: str, harness_id: str | None = None) -> dict[str, Any] | None:
        method = normalize_method(method)
        matched = self.metrics[
            (self.metrics["task_id"].astype(str) == str(task_id))
            & (self.metrics["method"].astype(str).str.upper() == method)
        ]

        if harness_id is not None and 'harness_id' in matched:
            matched=matched.loc[matched.harness_id.astype(str).eq(str(harness_id))]
        if matched.empty:
            return None

        return matched.iloc[-1].to_dict()

    def get_report_row(self, task_id: str, method: str, harness_id: str | None = None) -> dict[str, Any] | None:
        if harness_id is not None:
            return next((row for row in reversed(self.reports) if str(row.get('task_id'))==str(task_id)
                and normalize_method(row.get('method',''))==normalize_method(method)
                and str(row.get('harness_id',row.get('candidate_id','')))==str(harness_id)),None)
        return self._report_index.get((str(task_id), normalize_method(method)))

    def get_report(self, task_id: str, method: str, harness_id: str | None = None) -> dict[str, Any] | None:
        row = self.get_report_row(task_id, method,harness_id)
        if row is None:
            return None
        return row.get("report")

    def get_trace_row(self, task_id: str, method: str, harness_id: str | None = None) -> dict[str, Any] | None:
        if harness_id is not None:
            return next((row for row in reversed(self.traces) if str(row.get('task_id'))==str(task_id)
                and normalize_method(row.get('method',''))==normalize_method(method)
                and str(row.get('harness_id',row.get('candidate_id','')))==str(harness_id)),None)
        return self._trace_index.get((str(task_id), normalize_method(method)))

    def get_execution_log(self, task_id: str, method: str) -> dict[str, Any]:
        trace = self.get_trace_row(task_id, method)
        if trace is None:
            return {}
        return trace.get("execution_log", {}) or {}

    def get_raw_response(self, task_id: str, method: str) -> Any | None:
        trace = self.get_trace_row(task_id, method)
        if trace is None:
            return None
        return trace.get("raw_response")

    def get_failures_for_task_method(
        self,
        task_id: str,
        method: str,
    ) -> list[str]:
        metric_row = self.get_metric_row(task_id, method)
        if metric_row is None:
            return []

        failures: list[str] = []
        for flag in FAILURE_FLAGS:
            if int(metric_row.get(flag, 0) or 0) == 1:
                failures.append(flag)

        return failures

    # --------------------------------------------------------
    # Bundle construction
    # --------------------------------------------------------

    def build_trace_bundle(
        self,
        *,
        task_id: str,
        method: str,
        include_report: bool = True,
        harness_id: str | None = None,
    ) -> TraceBundle:
        method = normalize_method(method)

        metric_row = self.get_metric_row(task_id, method,harness_id)
        if metric_row is None:
            raise KeyError(f"Metric row not found for task_id={task_id}, method={method}")

        trace_row = self.get_trace_row(task_id, method,harness_id) or {}
        report = self.get_report(task_id, method,harness_id) if include_report else None

        failure_ids = [flag for flag in FAILURE_FLAGS if int(metric_row.get(flag,0) or 0)==1]
        failure_categories = [
            FAILURE_CATEGORIES.get(fid, "Unknown")
            for fid in failure_ids
        ]

        violation_context = self._make_violation_context(metric_row)

        reference=None
        if method=='M4':
            baseline=self.get_metric_row(task_id,'M3')
            if baseline is not None:
                reference=dict(method='M3',report=self.get_report(task_id,'M3'),
                    label_correct=baseline.get('label_correct'),
                    report_contract_version=baseline.get('report_contract_version'))
        return TraceBundle(
            task_id=str(task_id),
            dataset=str(metric_row.get("dataset", "")),
            split=str(trace_row.get("split", "")),
            method=method,
            harness_id=str(metric_row.get("harness_id", trace_row.get("harness_id", ""))),
            failure_ids=failure_ids,
            failure_categories=failure_categories,
            gold_label=_none_if_nan(metric_row.get("gold_label")),
            predicted_label=_none_if_nan(metric_row.get("predicted_label")),
            report=report,
            metric_row=metric_row,
            execution_log=trace_row.get("execution_log", {}) or {},
            violation_context=violation_context,
            raw_response_available=trace_row.get("raw_response") is not None,
            M3_reference=reference,
            initial_report=initial_report_from_trace(trace_row),
            action_filter_audit=(trace_row.get('raw_response') or {}).get('action_filter'),
        )

    def build_failure_bundles(
        self,
        *,
        methods: Iterable[str] | None = None,
        datasets: Iterable[str] | None = None,
        task_ids: Iterable[str] | None = None,
        failure_ids: Iterable[str] | None = None,
        max_bundles: int | None = None,
        include_report: bool = True,
        prioritize: bool = True,
    ) -> list[TraceBundle]:
        """
        Build compact trace bundles for failed task-method pairs.

        Multiple failure rows for the same task/method are collapsed into one
        TraceBundle containing all failure flags from metrics_by_task.
        """
        failure_df = self.filter_metrics(
            methods=methods,
            datasets=datasets,
            task_ids=task_ids,
            failure_ids=failure_ids,
            only_failed=True,
        )

        if failure_df.empty:
            return []

        identity=['task_id','method']+(['harness_id'] if 'harness_id' in failure_df else [])
        unique_pairs = (
            failure_df[identity]
            .drop_duplicates()
            .copy()
        )

        if prioritize:
            unique_pairs["priority_score"] = unique_pairs.apply(
                lambda row: self._priority_score(
                    task_id=str(row["task_id"]),
                    method=str(row["method"]),
                    preferred_failure_ids=failure_ids,
                    harness_id=str(row['harness_id']) if 'harness_id' in row else None,
                ),
                axis=1,
            )
            unique_pairs = unique_pairs.sort_values(
                by=["priority_score", "task_id", "method"],
                ascending=[False, True, True],
            )

        if max_bundles is not None:
            unique_pairs = unique_pairs.head(max_bundles)

        bundles: list[TraceBundle] = []

        for _, row in unique_pairs.iterrows():
            bundles.append(
                self.build_trace_bundle(
                    task_id=str(row["task_id"]),
                    method=str(row["method"]),
                    include_report=include_report,
                    harness_id=str(row['harness_id']) if 'harness_id' in row else None,
                )
            )

        return bundles

    def _priority_score(
        self,
        *,
        task_id: str,
        method: str,
        preferred_failure_ids: Iterable[str] | None = None,
        harness_id: str | None = None,
    ) -> float:
        metric_row = self.get_metric_row(task_id, method, harness_id)
        if metric_row is None:
            return 0.0

        if preferred_failure_ids is None:
            priority_order = DEFAULT_FAILURE_PRIORITY
        else:
            priority_order = [normalize_failure_id(f) for f in preferred_failure_ids]

        score = 0.0

        for idx, failure_id in enumerate(priority_order):
            if int(metric_row.get(failure_id, 0) or 0) == 1:
                # Earlier failure IDs in priority_order receive larger weight.
                score += float(len(priority_order) - idx)

        # Add continuous severity terms.
        score += float(metric_row.get("unsupported_claim_rate", 0.0) or 0.0)
        score += float(metric_row.get("unsafe_recommendation_rate", 0.0) or 0.0)
        score += float(metric_row.get("generic_action_rate", 0.0) or 0.0)
        score += float(metric_row.get("repair_count", 0.0) or 0.0) * 0.25

        return score

    @staticmethod
    def _make_violation_context(metric_row: dict[str, Any]) -> dict[str, Any]:
        failure_ids = [
            flag for flag in FAILURE_FLAGS
            if int(metric_row.get(flag, 0) or 0) == 1
        ]

        return {
            "failure_ids": failure_ids,
            "failure_categories": [
                FAILURE_CATEGORIES.get(flag, "Unknown")
                for flag in failure_ids
            ],
            "label_correct": metric_row.get("label_correct"),
            "evidence_precision": metric_row.get("evidence_precision"),
            "evidence_recall": metric_row.get("evidence_recall"),
            "unsupported_claim_rate": metric_row.get("unsupported_claim_rate"),
            "unsafe_recommendation_rate": metric_row.get("unsafe_recommendation_rate"),
            "generic_action_rate": metric_row.get("generic_action_rate"),
            "source_hallucination_rate": metric_row.get("source_hallucination_rate"),
            "temporal_violation_rate": metric_row.get("temporal_violation_rate"),
            "causal_overclaim_rate": metric_row.get("causal_overclaim_rate"),
            "repair_count": metric_row.get("repair_count", 0),
            "repair_failed": metric_row.get("repair_failed", False),
            "cost_per_report": metric_row.get("cost_per_report"),
        }

    # --------------------------------------------------------
    # Export helpers
    # --------------------------------------------------------

    def export_failure_bundles_jsonl(
        self,
        output_path: str | Path,
        *,
        methods: Iterable[str] | None = None,
        datasets: Iterable[str] | None = None,
        failure_ids: Iterable[str] | None = None,
        max_bundles: int | None = None,
        include_report: bool = True,
    ) -> None:
        bundles = self.build_failure_bundles(
            methods=methods,
            datasets=datasets,
            failure_ids=failure_ids,
            max_bundles=max_bundles,
            include_report=include_report,
        )

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as f:
            for bundle in bundles:
                f.write(json.dumps(bundle.to_dict(), ensure_ascii=False, default=str) + "\n")

    def export_summary_json(self, output_path: str | Path) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(self.summary(), f, ensure_ascii=False, indent=2, default=str)


# ============================================================
# Misc helpers
# ============================================================

def _none_if_nan(value: Any) -> Any | None:
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


# ============================================================
# Convenience loader
# ============================================================

def load_trace_repository(
    root_dir: str | Path,
    *,
    load_raw_traces: bool = True,
) -> TraceRepository:
    return TraceRepository(root_dir, load_raw_traces=load_raw_traces)


# ============================================================
# CLI smoke check
# ============================================================

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Inspect a search trace repository.")
    parser.add_argument(
        "--root-dir",
        required=True,
        help="Search output directory, e.g. outputs/search/api_small_test",
    )
    parser.add_argument(
        "--export-failure-bundles",
        default=None,
        help="Optional output JSONL path for compact failure bundles.",
    )
    parser.add_argument(
        "--max-bundles",
        type=int,
        default=10,
        help="Maximum failure bundles to export or preview.",
    )
    parser.add_argument(
        "--failure-ids",
        default=None,
        help="Comma-separated failure IDs to filter, e.g. F7,F10.",
    )

    args = parser.parse_args()

    repo = load_trace_repository(args.root_dir)

    print("=== Trace Repository Summary ===")
    for key, value in repo.summary().items():
        print(f"{key}: {value}")

    failure_ids = None
    if args.failure_ids:
        failure_ids = [
            normalize_failure_id(f)
            for f in args.failure_ids.split(",")
            if f.strip()
        ]

    bundles = repo.build_failure_bundles(
        failure_ids=failure_ids,
        max_bundles=args.max_bundles,
        include_report=True,
    )

    print("\n=== Failure Bundle Preview ===")
    print(f"n_bundles: {len(bundles)}")

    for bundle in bundles[:3]:
        print(
            {
                "task_id": bundle.task_id,
                "dataset": bundle.dataset,
                "method": bundle.method,
                "failure_ids": bundle.failure_ids,
                "gold_label": bundle.gold_label,
                "predicted_label": bundle.predicted_label,
                "unsafe": bundle.violation_context.get("unsafe_recommendation_rate"),
                "unsupported": bundle.violation_context.get("unsupported_claim_rate"),
                "repair_failed": bundle.violation_context.get("repair_failed"),
            }
        )

    if args.export_failure_bundles:
        repo.export_failure_bundles_jsonl(
            args.export_failure_bundles,
            failure_ids=failure_ids,
            max_bundles=args.max_bundles,
            include_report=True,
        )
        print(f"\nExported failure bundles: {args.export_failure_bundles}")


if __name__ == "__main__":
    main()
