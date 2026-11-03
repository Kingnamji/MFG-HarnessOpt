"""
Deterministic evidence matching.

This module compares generated evidence claims against preprocessed proxy
evidence tables. It supports the following body metrics:

- Evidence Precision
- Evidence Recall
- Source Hallucination Rate
- Temporal Violation Rate
- Causal Overclaim Rate
- F2--F6 failure traces

No LLM judge is used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from .observable_matching import values_equal, times_equal

from .structured_report_schema import (
    canonicalize_field,
    normalize_label,
    normalize_source,
    normalize_text,
)


@dataclass(frozen=True)
class EvidenceMatchResult:
    generated_count: int
    proxy_required_count: int
    matched_generated_ids: list[str]
    matched_proxy_ids: list[str]
    invalid_source_or_field_ids: list[str]
    value_contradiction_ids: list[str]
    temporal_violation_ids: list[str]
    causal_overclaim_ids: list[str]
    unmatched_generated_ids: list[str]

    @property
    def precision(self) -> float:
        if self.generated_count == 0:
            return 0.0
        return len(set(self.matched_generated_ids)) / self.generated_count

    @property
    def recall(self) -> float | None:
        if self.proxy_required_count == 0:
            return None
        return min(len(set(self.matched_proxy_ids)) / self.proxy_required_count, 1.0)


def records_to_dicts(records: Any) -> list[dict[str, Any]]:
    """Accept list[dict] or pandas DataFrame."""
    if records is None:
        return []

    if hasattr(records, "to_dict"):
        return records.to_dict("records")

    return list(records)


def is_context_support(value: Any) -> bool:
    return normalize_text(value) in {"context", "metadata", "general_context"}


def is_source_field_valid(
    evidence: dict[str, Any],
    allowed_schema: dict[str, Any],
    alias_map: dict[str, list[str]] | None = None,
) -> bool:
    allowed_sources = allowed_schema.get("allowed_sources", {})

    source = normalize_source(evidence.get("source", ""))
    field = canonicalize_field(evidence.get("field", ""), alias_map)

    if source not in allowed_sources:
        return False

    allowed_fields = {
        canonicalize_field(f, alias_map)
        for f in allowed_sources[source]
    }

    return field in allowed_fields


def detect_causal_overclaim(
    evidence: dict[str, Any],
    allowed_schema: dict[str, Any],
    reference_role: str | None = None,
) -> bool:
    """
    Detect contextual evidence being presented as causal.

    For Azure PdM, maintenance and machines are contextual by default.
    This rule should be specified in allowed_schema["causal_overclaim_rule"].
    """
    source = normalize_source(evidence.get("source", ""))
    role = normalize_text(evidence.get("role", ""))

    rule = allowed_schema.get("causal_overclaim_rule", {})
    contextual_sources = {
        normalize_source(src)
        for src in rule.get("contextual_sources", [])
    }

    # F6 is a structured-role check, not a natural-language causality judge.
    # A record-level role covers mixed sources (e.g. AI4I type metadata).
    contextual = source in contextual_sources or normalize_text(reference_role) == "contextual"
    return contextual and role == "causal"


def compatible_supports(
    generated_supports: Any,
    proxy_supports: Any,
    include_context_for_precision: bool,
) -> bool:
    gen = normalize_label(generated_supports)
    proxy = normalize_label(proxy_supports)

    # A fact may be cited merely as context even when the target also gives it
    # diagnostic relevance. The converse is not valid: contextual target data
    # cannot support an arbitrary asserted fault label.
    if is_context_support(generated_supports):
        return include_context_for_precision
    return not is_context_support(proxy_supports) and gen == proxy


def compatible_window(generated_window: Any, proxy_window: Any) -> bool:
    gen = normalize_text(generated_window)
    proxy = normalize_text(proxy_window)

    if gen == proxy:
        return True

    if proxy == "static_instance" and gen in {"static", "instance", "static_instance"}:
        return True

    return False


def compatible_direction(generated_direction: Any, proxy_direction: Any) -> bool:
    gen = normalize_text(generated_direction)
    proxy = normalize_text(proxy_direction)

    if gen == proxy:
        return True

    if proxy == "out_of_range" and gen in {"out_of_range", "abnormal", "high", "low"}:
        return True

    return False


def compatible_role(generated_role: Any, proxy_role: Any) -> bool:
    gen = normalize_text(generated_role)
    proxy = normalize_text(proxy_role)

    if gen == proxy:
        return True

    # Conservative compatibility. Do not allow contextual -> causal.
    if proxy == "symptomatic" and gen == "symptomatic":
        return True

    if proxy == "contextual" and gen == "contextual":
        return True

    if proxy == "exclusionary" and gen == "exclusionary":
        return True

    return False


def event_time_compatible(
    generated_event_time: Any,
    proxy_event_time: Any,
    generated_window: Any,
    proxy_window: Any,
) -> bool:
    """
    If event_time is provided by the report, it must match the proxy event time.
    If event_time is omitted, we rely on window compatibility.
    """
    return compatible_window(generated_window, proxy_window) and times_equal(generated_event_time, proxy_event_time)


def match_generated_evidence(
    generated_evidence: list[dict[str, Any]],
    proxy_evidence: Any,
    allowed_schema: dict[str, Any],
    alias_map: dict[str, list[str]] | None = None,
    include_context_for_precision: bool = True,
    include_context_in_recall: bool = False,
) -> EvidenceMatchResult:
    """
    Match generated evidence against proxy evidence.

    Important experimental convention:
    - Context evidence can count for precision if it is valid.
    - Context evidence is excluded from the Evidence Recall denominator by default.
      This keeps Evidence Recall focused on label/component-supporting evidence.
    """
    proxy_rows = records_to_dicts(proxy_evidence)
    generated_evidence = generated_evidence or []

    proxy_required = [
        row for row in proxy_rows
        if include_context_in_recall or not is_context_support(row.get("supports", ""))
    ]

    matched_generated_ids: list[str] = []
    matched_proxy_ids: list[str] = []
    invalid_source_or_field_ids: list[str] = []
    value_contradiction_ids: list[str] = []
    temporal_violation_ids: list[str] = []
    causal_overclaim_ids: list[str] = []

    from collections import Counter
    counts = Counter(str(e.get("evidence_id", f"GEN_{i+1}")) for i,e in enumerate(generated_evidence))
    unmatched_generated_ids = []
    for idx, generated in enumerate(generated_evidence):
        generated_id = str(generated.get("evidence_id", f"GEN_{idx + 1}"))

        if counts[generated_id] > 1 or not is_source_field_valid(generated, allowed_schema, alias_map):
            invalid_source_or_field_ids.append(generated_id)
            unmatched_generated_ids.append(generated_id)
            continue

        if detect_causal_overclaim(generated, allowed_schema):
            causal_overclaim_ids.append(generated_id)

        gen_source = normalize_source(generated.get("source", ""))
        gen_field = canonicalize_field(generated.get("field", ""), alias_map)

        matched_proxy = None
        candidates = []
        for proxy in proxy_rows:
            proxy_source = normalize_source(proxy.get("source", ""))
            proxy_field = canonicalize_field(proxy.get("field", ""), alias_map)

            if gen_source != proxy_source:
                continue

            if gen_field != proxy_field:
                continue

            candidates.append(proxy)
        failures = []
        for proxy in candidates:
            errors = set()
            if not compatible_supports(
                generated.get("supports", ""),
                proxy.get("supports", ""),
                include_context_for_precision=include_context_for_precision,
            ):
                errors.add("supports")

            if not compatible_window(generated.get("window", ""), proxy.get("window", "")):
                errors.add("time")

            if not compatible_direction(
                generated.get("direction", ""),
                proxy.get("direction", ""),
            ):
                errors.add("value")

            if not values_equal(generated.get("value"), proxy.get("value")):
                errors.add("value")

            if str(generated.get("statistic", "")).strip() != str(proxy.get("statistic", "")).strip():
                errors.add("value")

            if not compatible_role(generated.get("role", ""), proxy.get("role", "")):
                errors.add("role")

            if detect_causal_overclaim(generated, allowed_schema, proxy.get("role")):
                errors.add("causal")

            if not event_time_compatible(
                generated.get("event_time", None),
                proxy.get("event_time", None),
                generated.get("window", ""),
                proxy.get("window", ""),
            ):
                errors.add("time")
            if not errors:
                matched_proxy = proxy
                break
            failures.append((proxy, errors))

        if matched_proxy is not None:
            matched_generated_ids.append(generated_id)

            # Context evidence can count toward precision, but by default it should not
            # increase the Evidence Recall numerator because context evidence is excluded
            # from the recall denominator.
            same_asserted_support = normalize_label(generated.get("supports", "")) == normalize_label(matched_proxy.get("supports", ""))
            if same_asserted_support and (include_context_in_recall or not is_context_support(matched_proxy.get("supports", ""))):
                matched_proxy_ids.append(str(matched_proxy.get("evidence_id", "")))
        else:
            unmatched_generated_ids.append(generated_id)
            if not candidates:
                invalid_source_or_field_ids.append(generated_id)
            else:
                # Alternative records must not leave spurious flags when another
                # record fully matches. For failures, prefer the cited ID, then
                # the closest reference. M0/M1 may assign their own claim IDs.
                cited = [x for x in failures if str(x[0].get("evidence_id")) == generated_id]
                _, errors = min(cited or failures, key=lambda x: (len(x[1]), str(x[0].get("evidence_id", ""))))
                if "time" in errors:temporal_violation_ids.append(generated_id)
                if "value" in errors or ("role" in errors and "causal" not in errors):value_contradiction_ids.append(generated_id)
                if "causal" in errors:causal_overclaim_ids.append(generated_id)

    return EvidenceMatchResult(
        generated_count=len(generated_evidence),
        proxy_required_count=len(proxy_required),
        matched_generated_ids=sorted(set(matched_generated_ids)),
        matched_proxy_ids=sorted(set(matched_proxy_ids)),
        invalid_source_or_field_ids=sorted(set(invalid_source_or_field_ids)),
        value_contradiction_ids=sorted(set(value_contradiction_ids)),
        temporal_violation_ids=sorted(set(temporal_violation_ids)),
        causal_overclaim_ids=sorted(set(causal_overclaim_ids)),
        unmatched_generated_ids=sorted(set(unmatched_generated_ids)),
    )
