"""
Structured output schema for LLM-generated manufacturing diagnostic reports.

The schema is intentionally aligned with the deterministic evaluator:
- predicted_label
- confidence
- evidence[]
- next_steps[]
- limitations[]

This module contains no API calls. It only defines the JSON schema used by the
OpenAI client.
"""

from __future__ import annotations

from typing import Any


DIAGNOSTIC_REPORT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "predicted_label",
        "confidence",
        "evidence",
        "next_steps",
        "limitations",
    ],
    "properties": {
        "predicted_label": {
            "type": "string",
            "description": "Predicted fault mode, component, or No Failure label.",
        },
        "confidence": {
            "type": "string",
            "enum": ["low", "medium", "high"],
        },
        "evidence": {
            "type": "array",
            "description": "Structured evidence claims supporting the diagnosis.",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "evidence_id",
                    "source",
                    "field",
                    "statistic",
                    "direction",
                    "value",
                    "window",
                    "supports",
                    "role",
                ],
                "properties": {
                    "evidence_id": {
                        "type": "string",
                        "description": "Evidence ID such as E1, E2, E3.",
                    },
                    "source": {
                        "type": "string",
                        "description": "Data source name from the allowed source schema.",
                    },
                    "field": {
                        "type": "string",
                        "description": "Field or variable name from the allowed source schema.",
                    },
                    "statistic": {
                        "type": "string",
                        "description": "Copy the referenced observation's statistic exactly, e.g., observed_value or computed_value. This is the observation type, not direction or a measurement.",
                    },
                    "direction": {
                        "type": "string",
                        "description": "Copy the referenced observation's direction tag exactly. Preserve normal as normal; observed is allowed only when the record says observed. Never put a numeric measurement here.",
                    },
                    "value": {
                        "type": "string",
                        "description": "Copy the observation's numeric or categorical measurement into value as a string, not into statistic or direction.",
                    },
                    "window": {
                        "type": "string",
                        "description": "Relevant time window or static_instance.",
                    },
                    "supports": {
                        "type": "string",
                        "enum": ["No Failure", "TWF", "HDF", "PWF", "OSF", "comp1", "comp2", "comp3", "comp4", "context"],
                        "description": "Exactly one label from this task's candidate labels, or context. Put explanation, exclusions and negation in limitations; never append prose or multiple labels here.",
                    },
                    "role": {
                        "type": "string",
                        "description": "Copy the referenced observation's role exactly; diagnostic interpretation belongs in supports, not a changed role.",
                    },
                },
            },
        },
        "next_steps": {
            "type": "array",
            "description": "Evidence-linked next-step recommendations for human review.",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "action_type",
                    "target",
                    "rationale_evidence_ids",
                ],
                "properties": {
                    "action_type": {
                        "type": "string",
                        "description": "Action type from the action ontology.",
                    },
                    "target": {
                        "type": "string",
                        "description": "Specific target component, failure mode, sensor, or subsystem.",
                    },
                    "rationale_evidence_ids": {
                        "type": "array",
                        "items": {
                            "type": "string",
                        },
                        "description": "Evidence IDs that justify the action.",
                    },
                },
            },
        },
        "limitations": {
            "type": "array",
            "description": "Known uncertainty or missing information.",
            "items": {
                "type": "string",
            },
        },
    },
}


def get_diagnostic_report_json_schema() -> dict[str, Any]:
    return DIAGNOSTIC_REPORT_SCHEMA


def get_openai_text_format_schema() -> dict[str, Any]:
    """
    Return schema payload for OpenAI structured output.

    The OpenAI client wraps this inside the Responses API call.
    If the SDK changes, adjust openai_client.py only.
    """
    return {
        "type": "json_schema",
        "name": "manufacturing_diagnostic_report",
        "schema": DIAGNOSTIC_REPORT_SCHEMA,
        "strict": True,
    }
