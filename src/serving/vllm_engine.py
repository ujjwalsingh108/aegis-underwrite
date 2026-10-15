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

from typing import AsyncGenerator
import modal
from src.common.config import settings
from src.serving.modal_app import app, app_volume, cpu_image, gpu_image

# Dynamic compute configuration based on settings.use_gpu
active_image = gpu_image if settings.use_gpu else cpu_image
active_gpu = settings.gpu_type if settings.use_gpu else None

@app.cls(
    image=active_image,
    gpu=active_gpu,
    volumes={"/vol": app_volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    scaledown_window=60,
)

class UnderwritingLLMEngine:
    """Serverless worker class running on CPU or GPU based on configuration."""

    @modal.enter()
    def initialize_modal(self):
        print(f"Initializing model on {'GPU: ' + str(active_gpu) if settings.use_gpu else 'CPU'}")
        print(f"Target model identifier: {settings.base_model_name}")

        if settings.use_gpu:        
            from vllm import AsyncEngineArgs, AsyncLLMEngine

            engine_args = AsyncEngineArgs(
                model=settings.base_model_name,  # specifies which model to load
                tensor_parallel_size=1,  # splitting model layers accross multiple GPUs, 1 means no parallelism - the entire modal runs on single GPU
                gpu_memory_utilization=0.90,  # 90% of available GPU memory can be allocated for model weights, KV cache and inference
                max_model_len=4096,  # max context length(number of tokens model can handle in single prompt), 4096 tokens = ~3000 words
                enforce_eager=False,  # eager mode -> executes operations immediately(good for debugging, slower), Lazy/graph mode -> builds a computation graph and optimizes execution (faster). False means lazy execution is allowed, improving performance
                trust_remote_code=True,  # allows loading custom model code from external source(e.g., Hugging face models with custom layers)
            )

            self.engine = AsyncLLMEngine.from_engine_args(
                engine_args
            )  # enables non-blocking inference(multiple requests can be served concurrently)

        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            self.tokenizer = AutoTokenizer.from_pretrained(settings.base_model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                settings.base_model_name,
                torch_dtype=torch.float32,
                device_map="cpu",
                low_cpu_mem_usage=True
            )

    @modal.method()
    async def generate_stream(
        self,
        prompt: str,
        temperature: float = 0.1,  # controls randomness in generation(low = deterministic, hight = creative)
        max_tokens: int = 1024,
    ) -> AsyncGenerator[
        str, None
    ]:  # Function yields strings incrementally instead of returning one big string.
        """Generate streaming token responses for real-time synthesis."""

        if settings.use_gpu:
            import uuid
            from vllm import SamplingParams

            sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens)

            request_id = str(uuid.uuid4())
            results_generator = self.engine.generate(prompt, sampling_params, request_id)

            previous_text = ""

            async for request_output in results_generator:
                text = request_output.outputs[0].text
                delta = text[len(previous_text) :]
                previous_text = text
                yield delta

        else:
            # CPU token streaming via Hugging Face TextIteratorStreamer
            from threading import Thread
            from transformers import TextIteratorStreamer

            inputs = self.tokenizer(prompt, return_tensors="pt")
            streamer = TextIteratorStreamer(
                self.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True
            )

            generation_kwargs = dict(
                **inputs,
                streamer=streamer,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0.0,
            )

            thread = Thread(
                target=self.model.generate, # type: ignore
                kwargs=generation_kwargs
            )

            for new_text in streamer:
                yield new_text

@app.local_entrypoint()
def test_stream():
    """Real sample execution that streams tokens directly into your local terminal"""
    print("\n--- Connecting to Modal remote GPU worker ---")

    sample_prompt = (
        "<|im_start|>system\n"
        "You are an underwriting risk analyst. Summarize this credit note concisely.<|im_end|>\n"
        "<|im_start|>user\n"
        "Applicant: TechCorp Solutions.\n"
        "Stated monthly income: $50,000.\n"
        "Recurring debt obligations: $35,000.\n"
        "Explain the risk in two sentences.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    print(f"Sending prompt to cloud GPU...\n")
    print("=" * 20, "STREAMING OUTPUT", "=" * 20)

    engine = UnderwritingLLMEngine()

    import sys

    for token_delta in engine.generate_stream.remote_gen(sample_prompt):
        # Flush each token chunk directly to standard output without newline delays
        sys.stdout.write(token_delta)
        sys.stdout.flush()

    print("\n" + "=" * 58 + "\n")

