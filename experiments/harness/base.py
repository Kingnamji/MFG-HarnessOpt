"""
Base harness interface for manufacturing diagnostic report generation.

At this stage, harnesses can run in two modes:

1. Dummy mode:
   - No GPT API call.
   - Builds a structured diagnostic report from available task/proxy evidence.
   - Used for pipeline testing.

2. LLM mode:
   - A future llm_client can be injected.
   - The harness builds a prompt/context and delegates report generation.
   - This file already defines the expected interface, but does not depend on OpenAI.

All M0--M3 harnesses should return HarnessResult with the same fields.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd
from experiments.data.generation_evidence import generation_task, visible_evidence


StructuredReport = dict[str, Any]


@dataclass(frozen=True)
class HarnessResult:
    method: str
    harness_id: str
    task_id: str
    dataset: str

    report: StructuredReport

    accessed_sources: list[str] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)

    repair_count: int = 0
    repair_failed: bool = False

    latency_sec: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    model_name: str = "dummy"

    raw_response: Any | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def execution_log(self) -> dict[str, Any]:
        protocol=self.extra.get('protocol')
        if protocol is None and self.model_name=='dummy':
            from experiments.llm.report_contract import REPORT_CONTRACT_VERSION
            protocol={'report_contract_version':REPORT_CONTRACT_VERSION,'mode':'dummy'}
        result = {
            "task_id": self.task_id,
            "method": self.method,
            "harness_id": self.harness_id,
            "dataset": self.dataset,
            "accessed_sources": self.accessed_sources,
            "source_access_count": len(set(self.accessed_sources)),
            "tool_calls": self.tool_calls,
            "tool_call_count": len(self.tool_calls),
            "repair_count": self.repair_count,
            "repair_failed": self.repair_failed,
            "latency_sec": self.latency_sec,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": self.cost_usd,
            "model_name": self.model_name,
            "protocol": protocol,
        }
        action_filter = self.extra.get('action_filter')
        if action_filter is not None:
            result.update(action_filter_version=action_filter['version'],
                action_filter_removed_count=action_filter.get('removed_count', 0),
                action_filter_downgraded_count=action_filter.get('downgraded_count', 0),
                repair_attempt_failed=self.extra.get('repair_attempt_failed'))
        return result

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["execution_log"] = self.execution_log()
        return row


class DiagnosticHarness(ABC):
    """
    Abstract base class for M0--M4 diagnostic harnesses.
    """

    method: str = "BASE"
    harness_id: str = "base"

    def __init__(
        self,
        *,
        llm_client: Any | None = None,
        config: dict[str, Any] | None = None,
        dummy_mode: bool = True,
    ) -> None:
        self.llm_client = llm_client
        self.config = config or {}
        self.dummy_mode = dummy_mode

    @abstractmethod
    def run(
        self,
        *,
        task: dict[str, Any],
        proxy_evidence: pd.DataFrame | list[dict[str, Any]] | None = None,
        allowed_schema: dict[str, Any] | None = None,
        action_ontology: dict[str, Any] | None = None,
        action_preconditions: dict[str, Any] | None = None,
        generic_action_rules: dict[str, Any] | None = None,
    ) -> HarnessResult:
        raise NotImplementedError

    def _start_timer(self) -> float:
        return time.perf_counter()

    def _elapsed(self, start: float) -> float:
        return time.perf_counter() - start


# ============================================================
# Shared helper functions
# ============================================================

def records_to_dataframe(records: pd.DataFrame | list[dict[str, Any]] | None) -> pd.DataFrame:
    if records is None:
        return pd.DataFrame()

    if isinstance(records, pd.DataFrame):
        return records.copy()

    return pd.DataFrame(records)


def safe_value(value: Any) -> Any:
    if pd.isna(value):
        return ""
    return value


def proxy_row_to_evidence(row: pd.Series, evidence_id: str | None = None) -> dict[str, Any]:
    evidence = {
        "evidence_id": evidence_id or str(row.get("evidence_id", "")),
        "source": safe_value(row.get("source", "")),
        "field": safe_value(row.get("field", "")),
        "statistic": safe_value(row.get("statistic", "")),
        "direction": safe_value(row.get("direction", "")),
        "value": safe_value(row.get("value", "")),
        "window": safe_value(row.get("window", "")),
        "supports": "context",
        "role": safe_value(row.get("role", "")),
    }

    if "event_time" in row.index:
        evidence["event_time"] = safe_value(row.get("event_time", ""))

    return evidence


def select_proxy_evidence(
    *,
    task: dict[str, Any],
    proxy_evidence: pd.DataFrame | list[dict[str, Any]] | None,
    max_items: int,
    prefer_label_support: bool = True,
    include_context: bool = True,
) -> list[dict[str, Any]]:
    """
    Select evidence rows for dummy report construction.

    This is not used as the final LLM behavior. It is for pipeline testing and
    harness skeleton validation before GPT API integration.
    """
    df = visible_evidence(proxy_evidence)

    if df.empty:
        return []

    df = df.copy()
    if not include_context and "role" in df.columns:
        df = df[df["role"] != "contextual"].copy()

    if df.empty:
        return []

    selected = df.head(max_items)

    return [
        proxy_row_to_evidence(row, evidence_id=str(row.get("evidence_id", f"E{i + 1}")))
        for i, (_, row) in enumerate(selected.iterrows())
    ]


def build_structured_report(
    *,
    task: dict[str, Any],
    evidence: list[dict[str, Any]],
    action_type: str = "inspect",
    confidence: str = "medium",
    allow_no_evidence_action: bool = False,
) -> dict[str, Any]:
    # Deliberately trivial test output. Dummy runs never predict the answer
    # from a hidden target and are not evidence of diagnostic performance.
    predicted_label = "comp1" if "azure" in str(task.get("dataset", "")).lower() else "No Failure"
    rationale_ids = [e["evidence_id"] for e in evidence[:1]]

    if allow_no_evidence_action:
        rationale_ids = []

    return {
        "predicted_label": predicted_label,
        "confidence": confidence,
        "evidence": evidence,
        "next_steps": [
            {
                "action_type": action_type,
                "target": predicted_label,
                "rationale_evidence_ids": rationale_ids,
            }
        ],
        "limitations": [],
    }


def summarize_task_for_prompt(task: dict[str, Any], max_fields: int = 12) -> str:
    """
    Compact task summary for future prompt construction.
    """
    items = []

    for key, value in generation_task(task).items():
        if key in {"gold_label"}:
            continue

        items.append(f"{key}: {value}")

        if len(items) >= max_fields:
            break

    return "\n".join(items)


def summarize_evidence_for_prompt(evidence: pd.DataFrame | list[dict[str, Any]] | None, max_rows: int = 8) -> str:
    """
    Compact evidence table summary for future prompt construction.
    """
    df = records_to_dataframe(evidence)

    if df.empty:
        return "No proxy evidence is provided to the harness."

    cols = [
        col for col in [
            "evidence_id",
            "source",
            "field",
            "statistic",
            "direction",
            "value",
            "window",
            "role",
        ]
        if col in df.columns
    ]

    return df[cols].head(max_rows).to_string(index=False)
