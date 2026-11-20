"""
Single source of truth for the LoRA prompt format.

The adapter is trained on exactly this format (``local_dataset/*.jsonl``), so the serving path
MUST build prompts with the same functions or the model sees a different distribution at
inference time (train/serve skew). Training, serving and the smoke-test entrypoint all import
from here.
"""

from src.common.schemas import ApplicantProfile, ExtractedFinancialMetrics

SYSTEM_INSTRUCTION = (
    "You are an underwriting risk analyst. Review the verified applicant metrics, evaluate the "
    "financial ratios, and output the formal underwriting memo JSON."
)

MAX_DOCUMENT_EXCERPT_CHARS = 2000


def build_input_context(profile: ApplicantProfile, metrics: ExtractedFinancialMetrics) -> str:
    """User-turn content. Byte-for-byte identical to the format used in the training set."""
    return (
        f"Applicant ID: {profile.applicant_id}\n"
        f"Entity Type: {profile.entity_type.value}\n"
        f"Stated Monthly Income: ${profile.stated_monthly_income:,.2f}\n"
        f"Verified Monthly Income: ${metrics.verified_monthly_income:,.2f}\n"
        f"Total Monthly Debt Obligations: ${profile.total_monthly_debt:,.2f}\n"
        f"Liquid Reserves: ${profile.liquid_assets:,.2f}\n"
        f"Anomaly Flags: {metrics.anomaly_flags}"
    )


def to_chatml(instruction: str, input_context: str, completion: str | None = None) -> str:
    """Qwen ChatML. With ``completion`` it is a training row; without, an inference prompt."""
    prompt = (
        f"<|im_start|>system\n{instruction}<|im_end|>\n"
        f"<|im_start|>user\n{input_context}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    if completion is not None:
        prompt += f"{completion}<|im_end|>"
    return prompt


def build_inference_prompt(
    profile: ApplicantProfile,
    metrics: ExtractedFinancialMetrics,
    document_excerpt: str | None = None,
) -> str:
    context = build_input_context(profile, metrics)
    if document_excerpt:
        # Not present in the training set; appended last so the trained prefix is unchanged.
        context += f"\nDocument Excerpt:\n{document_excerpt[:MAX_DOCUMENT_EXCERPT_CHARS]}"
    return to_chatml(SYSTEM_INSTRUCTION, context)
