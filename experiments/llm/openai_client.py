"""
OpenAI client wrapper for manufacturing diagnostic report generation.

Responsibilities:
- load API key from environment/.env
- call OpenAI with structured output schema
- parse structured report
- log raw response, token usage, cost, latency, and model name

This file is intentionally isolated so SDK/API changes affect only one module.
"""

from __future__ import annotations
from experiments.utils.api_budget import current_api_budget

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

from experiments.llm.structured_output_schema import get_openai_text_format_schema

def load_openai_api_key(api_key_path: str | Path | None = None) -> str:
    """
    Load OpenAI API key from either:
    1. JSON file, if api_key_path is provided
    2. environment variable OPENAI_API_KEY
    3. .env file loaded by python-dotenv

    JSON format:
    {
      "OPENAI_API_KEY": "..."
    }
    """
    load_dotenv()

    if api_key_path is not None:
        path = Path(api_key_path)

        if not path.exists():
            raise FileNotFoundError(f"API key JSON file not found: {path}")

        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        api_key = data.get("OPENAI_API_KEY") or data.get("openai_api_key")

        if not api_key:
            raise KeyError(
                f"API key JSON must contain 'OPENAI_API_KEY' or 'openai_api_key': {path}"
            )

        return str(api_key)

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. Provide api_key_path or set OPENAI_API_KEY."
        )

    return api_key


@dataclass(frozen=True)
class LLMResult:
    parsed_report: dict[str, Any]
    raw_response: Any
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_sec: float
    model_name: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StructuredJSONResult:
    parsed_json: dict[str, Any] | None
    raw_response: Any
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_sec: float
    model_name: str
    error: str | None = None


class OpenAIClient:
    """
    Thin wrapper around the OpenAI Python SDK.

    Parameters
    ----------
    model:
        Model name for diagnosis calls.
    price_input_per_1m:
        Input token price per 1M tokens. Used only for local cost logging.
    price_output_per_1m:
        Output token price per 1M tokens. Used only for local cost logging.
    max_output_tokens:
        Maximum output tokens.
    temperature:
        Temperature for generation. Use 0 for reproducible experiments.
    dry_run:
        If True, no API call is made. A minimal dummy report is returned.
    """

    def __init__(
        self,
        *,
        model: str,
        price_input_per_1m: float = 0.0,
        price_output_per_1m: float = 0.0,
        max_output_tokens: int = 1200,
        temperature: float = 0.0,
        dry_run: bool = False,
        api_key_path: str | Path | None = None,
        sdk_client: Any | None = None,
        sdk_max_retries: int | None = None,
    ) -> None:
        self.model = model
        self.price_input_per_1m = price_input_per_1m
        self.price_output_per_1m = price_output_per_1m
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.dry_run = dry_run

        self._client = None

        if not self.dry_run:
            if sdk_client is not None:
                self._client = sdk_client
                return
            api_key = load_openai_api_key(api_key_path)

            if not api_key:
                raise EnvironmentError(
                    "OPENAI_API_KEY is not set."
                )

            from openai import OpenAI

            options = {} if sdk_max_retries is None else {'max_retries': sdk_max_retries}
            if current_api_budget() is not None:
                options['max_retries'] = 0
            self._client = OpenAI(api_key=api_key, **options)

    def _create_response(self, **kwargs):
        budget = current_api_budget()
        sdk = self._client
        if budget is not None:
            # Also covers a real SDK injected by a caller rather than constructed
            # above. Mocks and audited SDK facades need no with_options method.
            if hasattr(sdk, 'with_options'):
                sdk = sdk.with_options(max_retries=0)
            return budget.create(sdk.responses.create, **kwargs)
        return sdk.responses.create(**kwargs)

    def generate_structured_json(self, *, messages, text_format, reasoning_effort='none') -> StructuredJSONResult:
        """One SDK request, task-specific schema, no silent fallback or retry.

        Incomplete/refused/malformed responses retain their usage and raw body
        so rejected proposal costs remain auditable. The proposer disables SDK
        retries as well; diagnosis's existing retry behavior is unchanged.
        """
        if self.dry_run or self._client is None:
            raise RuntimeError('Structured patch generation requires an explicit real or injected client')
        start = time.perf_counter()
        response = self._create_response(
            model=self.model,input=messages,text={'format':text_format},
            max_output_tokens=self.max_output_tokens,temperature=self.temperature,
            reasoning={'effort':reasoning_effort},
        )
        raw = self._serialize_response(response)
        get = lambda k,default=None:raw.get(k,default) if isinstance(raw,dict) else getattr(response,k,default)
        input_tokens,output_tokens = self._extract_usage(response)
        error = None
        parsed = None
        status = get('status','completed')
        if status != 'completed' or get('error'):
            error = f"Response not completed: {status}; {get('incomplete_details') or get('error')}"
        elif any(block.get('type')=='refusal' for item in get('output',[]) or [] for block in item.get('content',[]) or []):
            error = 'Proposer response refused'
        else:
            try:
                parsed = self._extract_parsed_report(response)
                if not isinstance(parsed,dict):
                    raise ValueError('Patch output must be an object')
            except (ValueError,TypeError) as exc:
                error = f'Invalid structured JSON: {exc}'
        return StructuredJSONResult(parsed,raw,input_tokens,output_tokens,
            self.estimate_cost(input_tokens,output_tokens),time.perf_counter()-start,
            str(get('model',self.model)),error)


    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        input_cost = (input_tokens / 1_000_000) * self.price_input_per_1m
        output_cost = (output_tokens / 1_000_000) * self.price_output_per_1m
        return float(input_cost + output_cost)

    def _dummy_result(self, messages: list[dict[str, str]]) -> LLMResult:
        start = time.perf_counter()

        parsed_report = {
            "predicted_label": "No Failure",
            "confidence": "low",
            "evidence": [],
            "next_steps": [
                {
                    "action_type": "monitor",
                    "target": "machine",
                    "rationale_evidence_ids": [],
                }
            ],
            "limitations": [
                "Dry-run response. No API call was made."
            ],
        }

        return LLMResult(
            parsed_report=parsed_report,
            raw_response={
                "dry_run": True,
                "messages": messages,
            },
            input_tokens=0,
            output_tokens=0,
            cost_usd=0.0,
            latency_sec=time.perf_counter() - start,
            model_name=self.model,
        )

    @retry(
        wait=wait_exponential(multiplier=1, min=1, max=20),
        stop=lambda state: current_api_budget() is not None or state.attempt_number >= 3,
        reraise=True,
    )
    def generate_structured_report(
        self,
        *,
        messages: list[dict[str, str]],
    ) -> LLMResult:
        """
        Generate a structured diagnostic report.

        The client expects messages in:
        [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."}
        ]
        """
        if self.dry_run:
            return self._dummy_result(messages)

        if self._client is None:
            raise RuntimeError("OpenAI client is not initialized.")

        start = time.perf_counter()

        response = self._create_response(
            model=self.model,
            input=messages,
            text={
                "format": get_openai_text_format_schema(),
            },
            max_output_tokens=self.max_output_tokens,
            temperature=self.temperature,
        )

        latency_sec = time.perf_counter() - start

        parsed_report = self._extract_parsed_report(response)
        input_tokens, output_tokens = self._extract_usage(response)

        return LLMResult(
            parsed_report=parsed_report,
            raw_response=self._serialize_response(response),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=self.estimate_cost(input_tokens, output_tokens),
            latency_sec=latency_sec,
            model_name=self.model,
        )

    def _extract_parsed_report(self, response: Any) -> dict[str, Any]:
        """
        Extract structured report from the response.

        The OpenAI SDK may expose output text differently depending on version.
        This method tries multiple paths.
        """
        if hasattr(response, "output_text"):
            text = response.output_text
            if isinstance(text, str) and text.strip():
                return json.loads(text)

        if isinstance(response, dict):
            if "output_text" in response:
                return json.loads(response["output_text"])

        # Conservative fallback: inspect output blocks.
        output = getattr(response, "output", None)
        if output is None and isinstance(response, dict):
            output = response.get("output")

        if output:
            for item in output:
                content = getattr(item, "content", None)
                if content is None and isinstance(item, dict):
                    content = item.get("content")

                if not content:
                    continue

                for block in content:
                    # SDK object style.
                    text = getattr(block, "text", None)
                    if text:
                        return json.loads(text)

                    # Dict style.
                    if isinstance(block, dict):
                        if "text" in block and block["text"]:
                            return json.loads(block["text"])
                        if "json" in block and block["json"]:
                            return block["json"]

        raise ValueError(
            "Could not extract structured diagnostic report from OpenAI response."
        )

    def _extract_usage(self, response: Any) -> tuple[int, int]:
        usage = getattr(response, "usage", None)

        if usage is None and isinstance(response, dict):
            usage = response.get("usage")

        if usage is None:
            return 0, 0

        # Object style.
        input_tokens = getattr(usage, "input_tokens", None)
        output_tokens = getattr(usage, "output_tokens", None)

        if input_tokens is not None or output_tokens is not None:
            return int(input_tokens or 0), int(output_tokens or 0)

        # Dict style.
        if isinstance(usage, dict):
            return (
                int(usage.get("input_tokens", 0) or 0),
                int(usage.get("output_tokens", 0) or 0),
            )

        return 0, 0

    def _serialize_response(self, response: Any) -> Any:
        """
        Convert SDK response to a JSON-serializable object when possible.
        """
        if hasattr(response, "model_dump"):
            return response.model_dump()

        if hasattr(response, "dict"):
            return response.dict()

        if isinstance(response, dict):
            return response

        return str(response)


# ============================================================
# Convenience factory
# ============================================================

def create_openai_client_from_config(config: dict[str, Any]) -> OpenAIClient:
    return OpenAIClient(
        model=config["model"],
        price_input_per_1m=float(config.get("price_input_per_1m", 0.0)),
        price_output_per_1m=float(config.get("price_output_per_1m", 0.0)),
        max_output_tokens=int(config.get("max_output_tokens", 1200)),
        temperature=float(config.get("temperature", 0.0)),
        dry_run=bool(config.get("dry_run", False)),
        api_key_path=config.get("api_key_path"),
    )
