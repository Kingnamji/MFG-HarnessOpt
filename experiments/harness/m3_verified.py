"""
M3: Manually verified harness.

Intended final behavior:
- M2 source/evidence construction.
- Deterministic verification.
- Action-validity filtering.
- One-step repair if the report violates schema/evidence/action constraints.

Current skeleton behavior:
- In dummy mode, generates a report from proxy evidence and optionally runs the
  deterministic evaluator once.
- If the initial report has no valid action evidence, it repairs the action
  rationale by linking the first evidence item.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pandas as pd

from experiments.evaluation.runtime_verifier import verify_runtime_report
from experiments.harness.base import (
    DiagnosticHarness,
    HarnessResult,
    build_structured_report,
    records_to_dataframe,
    select_proxy_evidence,
)


class M3VerifiedHarness(DiagnosticHarness):
    method = "M3"
    harness_id = "M3_verified"

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

            repair_count = 0
            repair_failed = False

            # Optional deterministic verification and simple repair.
            if allowed_schema is not None and action_ontology is not None:
                initial_result = verify_runtime_report(
                    report=report,
                    generation_evidence=proxy_evidence,
                    allowed_schema=allowed_schema,
                    action_ontology=action_ontology,
                    action_preconditions=action_preconditions or {},
                    generic_action_rules=generic_action_rules or {},
                )

                if initial_result.get("F7", 0) == 1 and len(evidence) > 0:
                    repaired = deepcopy(report)
                    repaired["next_steps"] = [
                        {
                            "action_type": "inspect",
                            "target": str(report["predicted_label"]),
                            "rationale_evidence_ids": [evidence[0]["evidence_id"]],
                        }
                    ]
                    report = repaired
                    repair_count += 1

                final_result = verify_runtime_report(
                    report=report,
                    generation_evidence=proxy_evidence,
                    allowed_schema=allowed_schema,
                    action_ontology=action_ontology,
                    action_preconditions=action_preconditions or {},
                    generic_action_rules=generic_action_rules or {},
                )

                critical_flags = ["F5", "F6", "F7", "F8"]
                repair_failed = any(final_result.get(flag, 0) == 1 for flag in critical_flags)

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
                    {
                        "tool_name": "deterministic_verifier",
                        "status": "ok",
                    },
                    {
                        "tool_name": "action_validity_filter",
                        "status": "ok",
                    },
                ],
                repair_count=repair_count,
                repair_failed=repair_failed,
                latency_sec=self._elapsed(start),
                model_name="dummy",
            )

        raise NotImplementedError("LLM mode will be implemented after openai_client.py.")
