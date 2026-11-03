"""
M2: Tool/source-use harness.

Intended final behavior:
- Uses source policy and diagnostic evidence table.
- Allows manufacturing data-source access.
- No deterministic post-generation verifier or repair.

Current skeleton behavior:
- In dummy mode, exposes multiple proxy evidence items as if they were retrieved
  by source/tool access.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from experiments.harness.base import (
    DiagnosticHarness,
    HarnessResult,
    build_structured_report,
    records_to_dataframe,
    select_proxy_evidence,
)


class M2ToolSourceHarness(DiagnosticHarness):
    method = "M2"
    harness_id = "M2_tool_source"

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
        start = self._start_timer()

        if self.dummy_mode or self.llm_client is None:
            evidence = select_proxy_evidence(
                task=task,
                proxy_evidence=proxy_evidence,
                max_items=3,
                prefer_label_support=True,
                include_context=True,
            )

            evidence_df = records_to_dataframe(proxy_evidence)
            accessed_sources = []

            if not evidence_df.empty and "source" in evidence_df.columns:
                accessed_sources = sorted(set(evidence_df["source"].dropna().astype(str).tolist()))

            report = build_structured_report(
                task=task,
                evidence=evidence,
                action_type="inspect",
                confidence="high" if evidence else "medium",
                allow_no_evidence_action=(len(evidence) == 0),
            )

            return HarnessResult(
                method=self.method,
                harness_id=self.harness_id,
                task_id=str(task.get("task_id", "")),
                dataset=str(task.get("dataset", "")),
                report=report,
                accessed_sources=accessed_sources,
                tool_calls=[
                    {
                        "tool_name": "source_policy_lookup",
                        "status": "ok",
                    },
                    {
                        "tool_name": "diagnostic_evidence_table",
                        "status": "ok",
                    },
                ],
                repair_count=0,
                repair_failed=False,
                latency_sec=self._elapsed(start),
                model_name="dummy",
            )

        raise NotImplementedError("LLM mode will be implemented after openai_client.py.")