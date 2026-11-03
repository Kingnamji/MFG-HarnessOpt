"""
M1: Static diagnostic summary harness.

Intended final behavior:
- Uses compact static task summary and derived features.
- No dynamic source access.
- No post-generation verifier or repair.

Current skeleton behavior:
- In dummy mode, uses one label-supporting proxy evidence item as a static
  diagnostic summary proxy.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from experiments.harness.base import (
    DiagnosticHarness,
    HarnessResult,
    build_structured_report,
    select_proxy_evidence,
)


class M1StaticSummaryHarness(DiagnosticHarness):
    method = "M1"
    harness_id = "M1_static_summary"

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
                max_items=1,
                prefer_label_support=True,
                include_context=False,
            )

            report = build_structured_report(
                task=task,
                evidence=evidence,
                action_type="inspect",
                confidence="medium" if evidence else "low",
                allow_no_evidence_action=(len(evidence) == 0),
            )

            return HarnessResult(
                method=self.method,
                harness_id=self.harness_id,
                task_id=str(task.get("task_id", "")),
                dataset=str(task.get("dataset", "")),
                report=report,
                accessed_sources=["static_summary"] if evidence else [],
                tool_calls=[
                    {
                        "tool_name": "build_static_summary",
                        "status": "ok",
                    }
                ],
                repair_count=0,
                repair_failed=False,
                latency_sec=self._elapsed(start),
                model_name="dummy",
            )

        raise NotImplementedError("LLM mode will be implemented after openai_client.py.")