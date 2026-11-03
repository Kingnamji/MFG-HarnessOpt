"""
Harness registry.

This module centralizes M0--M3 harness construction so runners can instantiate
methods by method ID.
"""

from __future__ import annotations

from typing import Any

from experiments.harness.base import DiagnosticHarness
from experiments.harness.m0_zero_shot import M0ZeroShotHarness
from experiments.harness.m1_static_summary import M1StaticSummaryHarness
from experiments.harness.m2_tool_source import M2ToolSourceHarness
from experiments.harness.m3_verified import M3VerifiedHarness


HARNESS_REGISTRY = {
    "M0": M0ZeroShotHarness,
    "M1": M1StaticSummaryHarness,
    "M2": M2ToolSourceHarness,
    "M3": M3VerifiedHarness,
}


def list_available_harnesses() -> list[str]:
    return list(HARNESS_REGISTRY.keys())


def create_harness(
    method: str,
    *,
    llm_client: Any | None = None,
    config: dict[str, Any] | None = None,
    dummy_mode: bool = True,
) -> DiagnosticHarness:
    method = method.upper()

    if method not in HARNESS_REGISTRY:
        raise KeyError(
            f"Unknown harness method: {method}. "
            f"Available methods: {list_available_harnesses()}"
        )

    harness_cls = HARNESS_REGISTRY[method]

    return harness_cls(
        llm_client=llm_client,
        config=config or {},
        dummy_mode=dummy_mode,
    )


def create_harnesses(
    methods: list[str] | None = None,
    *,
    llm_client: Any | None = None,
    config: dict[str, Any] | None = None,
    dummy_mode: bool = True,
) -> dict[str, DiagnosticHarness]:
    methods = methods or list_available_harnesses()

    return {
        method.upper(): create_harness(
            method,
            llm_client=llm_client,
            config=config,
            dummy_mode=dummy_mode,
        )
        for method in methods
    }