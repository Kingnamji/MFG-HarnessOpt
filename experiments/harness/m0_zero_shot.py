"""
M0: Zero-shot structured LLM baseline.

Intended final behavior:
- Minimal diagnostic context.
- Structured output required.
- No tool/source policy.
- No deterministic verification or repair inside the harness.

Current skeleton behavior:
- In dummy mode, generates a minimal structured report without proxy evidence.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from experiments.harness.base import (
    DiagnosticHarness,
    HarnessResult,
    build_structured_report,
)


class M0ZeroShotHarness(DiagnosticHarness):
    method = "M0"
    harness_id = "M0_zero_shot"

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
            report = build_structured_report(
                task=task,
                evidence=[],
                action_type="inspect",
                confidence="medium",
                allow_no_evidence_action=True,
            )

            return HarnessResult(
                method=self.method,
                harness_id=self.harness_id,
                task_id=str(task.get("task_id", "")),
                dataset=str(task.get("dataset", "")),
                report=report,
                accessed_sources=[],
                tool_calls=[],
                repair_count=0,
                repair_failed=False,
                latency_sec=self._elapsed(start),
                model_name="dummy",
            )

        raise NotImplementedError("LLM mode will be implemented after openai_client.py.")