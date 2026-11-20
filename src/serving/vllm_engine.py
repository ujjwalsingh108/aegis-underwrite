"""
Encapsulates the vLLM engine inside a Modal GPU class with lifecycle hooks,
continuous batching, and prefix caching.

This class runs exclusively on an NVIDIA GPU in Modal.
"""

import json
import os
import uuid
from collections.abc import AsyncGenerator
from typing import Any

import modal

from src.common.config import settings
from src.common.logger import get_logger
from src.common.prompts import build_inference_prompt
from src.common.schemas import (
    ApplicantProfile,
    EntityType,
    ExtractedFinancialMetrics,
    MultimodalFraudMetrics,
    UnderwritingMemo,
)
from src.serving.modal_app import app, app_volume, gpu_image

logger = get_logger("vllm")


def _structured_output_kwargs(schema_str: str) -> dict[str, Any]:
    """Builds JSON-schema constrained decoding kwargs for whichever vLLM version is installed.

    vLLM <=0.5 has no per-request guided decoding on SamplingParams, 0.6-0.10 exposes
    ``GuidedDecodingParams``/``guided_decoding`` and newer releases ``StructuredOutputsParams``/
    ``structured_outputs``. Without support we decode unconstrained and rely on Pydantic
    validation in ``generate_memo``.
    """
    try:
        from vllm.sampling_params import StructuredOutputsParams  # type: ignore

        return {"structured_outputs": StructuredOutputsParams(json=schema_str)}
    except ImportError:
        pass
    try:
        from vllm.sampling_params import GuidedDecodingParams  # type: ignore

        return {"guided_decoding": GuidedDecodingParams(json=schema_str)}
    except ImportError:
        logger.warning("vLLM build has no per-request structured outputs; decoding unconstrained")
        return {}


@app.cls(
    image=gpu_image,
    gpu=settings.gpu_type,
    volumes={"/vol": app_volume},
    secrets=[modal.Secret.from_name("underwriting_hf_token")],
    timeout=600,
)
class UnderwritingLLMEngine:
    """Serverless worker class strictly utilizing GPU for vLLM inference."""

    @modal.enter()
    def initialize_modal(self):
        from vllm import AsyncEngineArgs, AsyncLLMEngine

        logger.info(
            "initializing_model",
            extra={"gpu": settings.gpu_type, "model": settings.base_model_name},
        )

        adapter_dir = settings.lora_output_dir
        self.has_lora = os.path.exists(os.path.join(adapter_dir, "adapter_config.json"))

        logger.info("lora_adapter", extra={"found": self.has_lora, "path": adapter_dir})

        engine_args = AsyncEngineArgs(
            model=settings.base_model_name,                 # type: ignore
            tensor_parallel_size=1,                         # type: ignore
            gpu_memory_utilization=0.90,                    # type: ignore    
            max_model_len=4096,                             # type: ignore
            enforce_eager=False,                            # type: ignore
            trust_remote_code=True,                         # type: ignore
            enable_lora=self.has_lora,                      # type: ignore                          
            # max_loras / max_lora_rank are only read when enable_lora=True
            max_loras=1,                                    # type: ignore
            max_lora_rank=16,                               # type: ignore
        )

        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

    async def _stream_internal(
        self,
        prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> AsyncGenerator[str, None]:
        """Internal generator containing actual streaming inference logic via vLLM."""
        from vllm import SamplingParams
        from vllm.lora.request import LoRARequest

        schema_str = json.dumps(UnderwritingMemo.model_json_schema())
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            **_structured_output_kwargs(schema_str),
        )

        lora_request = (
            LoRARequest("underwrite_lora", 1, settings.lora_output_dir)
            if self.has_lora else None
        )

        request_id = str(uuid.uuid4())
        results_generator = self.engine.generate(
            prompt, sampling_params, request_id, lora_request=lora_request
        )

        previous_text = ""
        async for request_output in results_generator:
            text = request_output.outputs[0].text
            delta = text[len(previous_text) :]
            previous_text = text
            yield delta

    @modal.method()
    async def generate_stream(
        self,
        prompt: str,
        temperature: float = 0.1, # the lower the temperature, the more deterministic the output
        max_tokens: int = 1024,
    ) -> AsyncGenerator[str, None]:
        """Exposed Modal method for external streaming consumption."""
        async for chunk in self._stream_internal(prompt, temperature, max_tokens):
            yield chunk

    @modal.method()
    async def generate_memo(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> UnderwritingMemo:
        """
        Executes inference and strictly validates generated text against
        the UnderwritingMemo Pydantic schema.
        """
        raw_text_chunks = [
            chunk async for chunk in self._stream_internal(prompt, temperature, max_tokens)
        ]
        full_output = "".join(raw_text_chunks).strip()

        # Strip Markdown code blocks if generated by LLM
        clean_json = full_output
        if clean_json.startswith("```json"):
            clean_json = clean_json.removeprefix("```json").removesuffix("```").strip()
        elif clean_json.startswith("```"):
            clean_json = clean_json.removeprefix("```").removesuffix("```").strip()

        try:
            parsed = json.loads(clean_json)
            if "underwriting_memo" in parsed:
                parsed = parsed["underwriting_memo"]
            return UnderwritingMemo.model_validate(parsed)
        except Exception as e:
            # Raw output can contain applicant PII: keep it in worker logs, not in the exception
            # that propagates to the gateway (and potentially to clients).
            logger.warning(
                "memo_schema_validation_failed",
                extra={"error": str(e)[:500], "raw_output": full_output[:2000]},
            )
            raise ValueError(f"LLM memo failed schema validation: {type(e).__name__}") from None

    @modal.exit()
    def cleanup_modal(self):
        """Gracefully shuts down the vLLM engine before the Modal container exits."""
        if hasattr(self, "engine") and hasattr(self.engine, "shutdown"):
            try:
                self.engine.shutdown()
            except Exception as e:
                logger.warning("engine_shutdown_notice", extra={"error": str(e)})


@app.local_entrypoint()
def test_stream():
    """Runs a remote memo generation with the canonical prompt and validates the schema."""
    profile = ApplicantProfile(
        applicant_id="APP-DEMO-01",
        entity_type=EntityType.INDIVIDUAL,
        requested_loan_amount=250_000.0,
        stated_monthly_income=50_000.0,
        total_monthly_debt=35_000.0,
        liquid_assets=10_000.0,
    )
    metrics = ExtractedFinancialMetrics(
        verified_monthly_income=50_000.0,
        detected_monthly_liabilities=35_000.0,
        verified_liquid_reserves=10_000.0,
        fraud_metrics=MultimodalFraudMetrics(kyc_confidence_score=0.98, document_tampering_score=0.01),
        income_variance_pct=0.0,
    )
    engine = UnderwritingLLMEngine()
    try:
        memo = engine.generate_memo.remote(build_inference_prompt(profile, metrics))
        logger.info(
            "smoke_test_ok",
            extra={
                "decision": memo.decision.value,
                "dti": memo.calculated_dti,
                "runway_months": memo.liquidity_runway_months,
            },
        )
    except Exception as exc:
        logger.error("smoke_test_failed", extra={"error": str(exc)})
