"""
src/ui/dashboard.py
Gradio underwriter console for the Aegis Underwriting API.

Run:  AEGIS_API_URL=https://<workspace>--aegis-underwrite-web-endpoint.modal.run \
      AEGIS_API_KEY=<underwriter key> python -m src.ui.dashboard
"""

import os

import gradio as gr
import requests

# The UI is a separate client process, so it reads its own two variables rather than the
# server's settings (it must never hold the server's admin key).
API_BASE_URL = os.environ.get("AEGIS_API_URL", "http://127.0.0.1:8000").rstrip("/")
API_KEY = os.environ.get("AEGIS_API_KEY", "")
TIMEOUT_SECONDS = 600


def _fmt_ratio(value, pct: bool) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, str):  # "Infinity"
        return value
    return f"{value:.2%}" if pct else f"{value:.2f}"


def analyze_risk(
    applicant_id, entity_type, loan_amount, stated_inc, verified_inc, debt, reserves,
    kyc_score, tampering_score, sanctions_hit, flags_text,
):
    if not API_KEY:
        return "ERROR", "N/A", "N/A", "Set AEGIS_API_KEY to an underwriter API key.", "", {}

    payload = {
        "applicant_id": applicant_id,
        "entity_type": entity_type,
        "requested_loan_amount": loan_amount,
        "stated_monthly_income": stated_inc,
        "verified_monthly_income": verified_inc,
        "total_monthly_debt": debt,
        "liquid_reserves": reserves,
        "kyc_confidence_score": kyc_score,
        "document_tampering_score": tampering_score,
        "pep_or_sanctions_hit": sanctions_hit,
        "anomaly_flags": [f.strip() for f in flags_text.split(",") if f.strip()],
    }
    try:
        response = requests.post(
            f"{API_BASE_URL}/underwrite",
            json=payload,
            headers={"X-API-Key": API_KEY},
            timeout=TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        record = response.json()
    except Exception as e:
        return "ERROR", "N/A", "N/A", f"API Request Failed: {e}", "", {}

    memo = record["memo"]
    reasons = "\n".join(record.get("decision_reasons", [])) or "None"
    breaches = "\n".join(memo.get("breached_policies", [])) or "None detected"
    return (
        f"{memo['decision']}  ({record.get('decision_rule')})",
        _fmt_ratio(memo.get("calculated_dti"), pct=True),
        f"{_fmt_ratio(memo.get('liquidity_runway_months'), pct=False)} months",
        f"Reasons:\n{reasons}\n\nBreaches:\n{breaches}",
        "\n".join(memo.get("risk_factors", [])) or "None noted",
        record,
    )


with gr.Blocks(theme=gr.themes.Soft(primary_hue="slate"), title="Aegis Underwriting") as demo:
    gr.Markdown("# Aegis Copilot: Automated Underwriting")
    gr.Markdown(f"Connected to `{API_BASE_URL}`")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Applicant Data")
            app_id = gr.Textbox(label="Applicant ID", value="APP-PROD-001")
            entity = gr.Dropdown(["INDIVIDUAL", "LLC", "CORP", "TRUST"], value="INDIVIDUAL",
                                 label="Entity Type")
            loan_amount = gr.Number(label="Requested Loan Amount ($)", value=250000.0)
            stated_inc = gr.Number(label="Stated Monthly Income ($)", value=60000.0)
            verified_inc = gr.Number(label="Verified Monthly Income ($)", value=60000.0)
            debt = gr.Number(label="Total Monthly Debt ($)", value=25000.0)
            reserves = gr.Number(label="Liquid Reserves ($)", value=180000.0)
            gr.Markdown("### Verification Results")
            kyc = gr.Slider(0, 1, value=0.97, step=0.01, label="KYC Confidence Score")
            tampering = gr.Slider(0, 1, value=0.02, step=0.01, label="Document Tampering Score")
            sanctions = gr.Checkbox(label="PEP / Sanctions Hit", value=False)
            flags = gr.Textbox(label="Anomaly Flags (comma-separated)",
                               placeholder="e.g. Missing tax transcript")
            submit_btn = gr.Button("Generate Risk Memo", variant="primary")

        with gr.Column(scale=2):
            gr.Markdown("### Underwriting Decision")
            with gr.Row():
                out_decision = gr.Textbox(label="Final Decision", scale=1)
                out_dti = gr.Textbox(label="Calculated DTI", scale=1)
                out_runway = gr.Textbox(label="Liquidity Runway", scale=1)
            with gr.Row():
                out_breaches = gr.Textbox(label="Reasons & Policy Breaches", lines=6, scale=1)
                out_factors = gr.Textbox(label="Key Risk Factors", lines=6, scale=1)
            with gr.Accordion("Raw Decision Record", open=False):
                out_json = gr.JSON()

    submit_btn.click(
        fn=analyze_risk,
        inputs=[app_id, entity, loan_amount, stated_inc, verified_inc, debt, reserves,
                kyc, tampering, sanctions, flags],
        outputs=[out_decision, out_dti, out_runway, out_breaches, out_factors, out_json],
    )


if __name__ == "__main__":
    demo.launch()
