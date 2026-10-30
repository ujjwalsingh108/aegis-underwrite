"""
Encapsulates the vLLM engine inside a Modal GPU class with lifecycle hooks,
continuous batching, and prefix caching.

This class runs on an NVIDIA A10G or L4 GPU in Modal. It uses:
    * @app.cls: Marks the class as a Modal remote worker with scale-to-zero
    capability.

    * @modal.enter(): A lifecycle hook that runs once when the container boots up. It pulls the
    modal weights into GPU VRAM and initializes AsyncLLMEngine.

    * @modal.method(): A remote callable method that takes a prompt and generation
    parameters, returning streamed or batch tokens.
"""

import os
import json
import sys
from typing import AsyncGenerator, Optional
import modal
import torch
from peft import PeftModel
from rich import print
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.common.config import settings
from src.common.schemas import UnderwritingMemo
from src.serving.modal_app import app, app_volume, cpu_image, gpu_image

# Dynamic compute configuration based on settings.use_gpu
active_image = gpu_image if settings.use_gpu else cpu_image
active_gpu = settings.gpu_type if settings.use_gpu else None


@app.cls(
    image=active_image,
    gpu=active_gpu,
    volumes={"/vol": app_volume},
    secrets=[modal.Secret.from_name("underwriting_hf_token")],
    timeout=600,
)
class UnderwritingLLMEngine:
    """Serverless worker class running on CPU or GPU based on configuration."""

    @modal.enter()
    def initialize_modal(self):
        print(
            f"Initializing model on {'GPU: ' + str(active_gpu) if settings.use_gpu else 'CPU'}"
        )
        print(f"Target model identifier: {settings.base_model_name}")

        adapter_dir = settings.lora_output_dir
        self.has_lora = os.path.exists(os.path.join(adapter_dir, "adapter_config.json"))

        if self.has_lora:
            print(f"Found trained LoRA adapter at {adapter_dir}")
        else:
            print(f"No LoRA adapter found at {adapter_dir}. Running base modal.")

        if settings.use_gpu:
            from vllm import AsyncEngineArgs, AsyncLLMEngine

            engine_args = AsyncEngineArgs(
                model=settings.base_model_name,  # specifies which model to load
                tensor_parallel_size=1,  # splitting model layers accross multiple GPUs, 1 means no parallelism - the entire modal runs on single GPU
                gpu_memory_utilization=0.90,  # 90% of available GPU memory can be allocated for model weights, KV cache and inference
                max_model_len=4096,  # max context length(number of tokens model can handle in single prompt), 4096 tokens = ~3000 words
                enforce_eager=False,  # eager mode -> executes operations immediately(good for debugging, slower), Lazy/graph mode -> builds a computation graph and optimizes execution (faster). False means lazy execution is allowed, improving performance
                trust_remote_code=True,  # allows loading custom model code from external source(e.g., Hugging face models with custom layers)
                enable_lora=self.has_lora,
                max_loras=1 if self.has_lora else None, # type: ignore
                max_lora_rank=16 if self.has_lora else None, #type: ignore
            )

            self.engine = AsyncLLMEngine.from_engine_args(
                engine_args
            )  # enables non-blocking inference(multiple requests can be served concurrently)

        else:
            self.tokenizer = AutoTokenizer.from_pretrained(settings.base_model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Use torch.bfloat16 or float16 on GPU, float32 on CPU
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

            self.model = AutoModelForCausalLM.from_pretrained(
                settings.base_model_name,
                dtype=model_dtype,
                device_map=device,
                low_cpu_mem_usage=True,
            )

            if self.has_lora:
                print(f"Applying LoRA adapter from {adapter_dir}...")
                self.modal = PeftModel.from_pretrained(self.model, adapter_dir)
            else:
                self.model

            self.model.eval()

    async def _stream_internal(
        self,
        prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> AsyncGenerator[str, None]:
        """Internal generator containing actual streaming inference logic."""
        if settings.use_gpu:
            import uuid
            from vllm import SamplingParams
            from vllm.lora.request import LoRARequest
            from vllm.sampling_params import GuidedDecodingParams # type: ignore

            # Force the LLM to strictly follow the Pydantic schema
            schema_str = json.dumps(UnderwritingMemo.model_json_schema())
            guided_params = GuidedDecodingParams(json=schema_str)

            sampling_params = SamplingParams(
                temperature=temperature, 
                max_tokens=max_tokens,
                guided_decoding=guided_params  # Inject the constraints here # type: ignore
            )

            lora_request = (
                LoRARequest("underwrite_lora", 1, settings.lora_output_dir)
                if self.has_lora
                else None
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

        else:
            from threading import Thread
            from transformers import TextIteratorStreamer

            inputs = self.tokenizer(prompt, return_tensors="pt")
            streamer = TextIteratorStreamer(
                self.tokenizer, skip_prompt=True, skip_special_tokens=True
            )

            generation_kwargs = dict(
                **inputs,
                streamer=streamer,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0.0,
            )

            # Use a lambda wrapper to avoid Pylance callable binding issues
            thread = Thread(
                target=lambda: self.model.generate(**generation_kwargs) # type: ignore
            )
            thread.start()

            for new_text in streamer:
                yield new_text

    @modal.method()
    async def generate_stream(
        self,
        prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> AsyncGenerator[str, None]:
        """Exposed Modal method for external streaming consumption."""
        async for chunk in self._stream_internal(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        ):
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
        raw_text_chunks = []
        async for chunk in self._stream_internal(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        ):
            raw_text_chunks.append(chunk)

        full_output = "".join(raw_text_chunks).strip()

        # Strip Markdown code blocks if generated by LLM
        clean_json = full_output
        if clean_json.startswith("```json"):
            clean_json = clean_json.removeprefix("```json").removesuffix("```").strip()
        elif clean_json.startswith("```"):
            clean_json = clean_json.removeprefix("```").removesuffix("```").strip()

        # Enforce Pydantic validation
        try:
            parsed = json.loads(clean_json)
            if "underwriting_memo" in parsed:
                parsed = parsed["underwriting_memo"]
            return UnderwritingMemo.model_validate(parsed)
        except Exception as e:
            raise ValueError(
                f"Model output failed UnderwritingMemo schema validation: {e}\nRaw output:\n{full_output}"
            )

    @modal.exit()
    def cleanup_modal(self):
        """Gracefully shuts down the vLLM engine before the Modal container exits."""
        if settings.use_gpu and hasattr(self, "engine"):
            try:
                if hasattr(self.engine, "shutdown"):
                    self.engine.shutdown()  # Synchronous call
            except Exception as e:
                print(f"Notice during shutdown: {e}")

@app.local_entrypoint()
def test_stream():
    """Runs token streaming and validates Pydantic JSON decoding."""
    print("\n--- Connecting to Modal remote GPU worker ---")
    engine = UnderwritingLLMEngine()

    test_prompt = (
        "<|im_start|>system\n"
        "You are an underwriting risk analyst. Review the verified applicant metrics, "
        "evaluate the financial ratios, and output the formal underwriting memo JSON "
        "with keys: application_id, decision, calculated_dti, calculated_dscr, "
        "liquidity_runway_months, breached_policies, risk_factors, covenants, audit_trail.<|im_end|>\n"
        "<|im_start|>user\n"
        "Applicant ID: APP-DEMO-01\n"
        "Stated Monthly Income: $50,000.00\n"
        "Verified Monthly Income: $50,000.00\n"
        "Total Monthly Debt Obligations: $35,000.00\n"
        "Liquid Reserves: $10,000.00\n"
        "Income Variance: 0.0%\n"
        "Anomaly Flags: ['DTI exceeds 43% policy threshold']<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    print("Executing streaming inference...\n")
    print("=" * 20, "STREAMING OUTPUT", "=" * 20)

    for token_delta in engine.generate_stream.remote_gen(test_prompt):
        sys.stdout.write(token_delta)
        sys.stdout.flush()
    print("\n" + "=" * 58 + "\n")

    print("Executing structured UnderwritingMemo schema validation...")

    try:
        memo = engine.generate_memo.remote(test_prompt)
        print("Schema validation successful!")
        print(f"Decision: {memo.decision}")
        print(f"Computed DTI: {memo.calculated_dti:.2%}")
        print(f"Liquidity Runway: {memo.liquidity_runway_months} months")
        print(f"Policy Breaches: {memo.breached_policies}")

    except Exception as exc:
        print(f"Schema validation check error: {exc}")
