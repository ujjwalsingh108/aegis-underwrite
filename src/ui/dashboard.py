"""
src/ui/dashboard.py
Gradio frontend for the Aegis Underwriting Pipeline.
"""
import gradio as gr
import requests
import json

# PASTE YOUR FASTAPI ENDPOINT HERE (keep the /underwrite path)
API_URL = "https://usingh2051--aegis-underwrite-api-dev.us-east.modal.run/underwrite"

def analyze_risk(applicant_id, stated_inc, verified_inc, debt, reserves, flags_text):
    if "YOUR-WORKSPACE-NAME" in API_URL:
        return "ERROR", "Update API_URL in src/ui/dashboard.py", "", "", "", {}
        
    flags = [f.strip() for f in flags_text.split(",") if f.strip()]
    
    payload = {
        "applicant_id": applicant_id,
        "stated_monthly_income": stated_inc,
        "verified_monthly_income": verified_inc,
        "total_monthly_debt": debt,
        "liquid_reserves": reserves,
        "anomaly_flags": flags
    }
    
    try:
        response = requests.post(API_URL, json=payload, timeout=300)
        response.raise_for_status()
        data = response.json()
        
        decision = data.get("decision", "UNKNOWN")
        dti = f"{data.get('calculated_dti', 0):.2f}%"
        runway = f"{data.get('liquidity_runway_months', 0)} months"
        breaches = "\n".join(data.get("breached_policies", [])) or "None detected"
        factors = "\n".join(data.get("risk_factors", [])) or "None noted"
        
        return decision, dti, runway, breaches, factors, data
        
    except Exception as e:
        return "ERROR", "N/A", "N/A", f"API Request Failed: {str(e)}", "", {}

# Build the Production-Grade UI Layout
with gr.Blocks(theme=gr.themes.Soft(primary_hue="slate"), title="Aegis Underwriting") as demo:
    gr.Markdown("# 🛡️ Aegis Copilot: Automated Underwriting")
    gr.Markdown("Enter the verified applicant financials below to generate an LLM-evaluated risk memo.")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Applicant Data")
            app_id = gr.Textbox(label="Applicant ID", value="APP-PROD-001")
            stated_inc = gr.Number(label="Stated Monthly Income ($)", value=60000.0)
            verified_inc = gr.Number(label="Verified Monthly Income ($)", value=60000.0)
            debt = gr.Number(label="Total Monthly Debt ($)", value=25000.0)
            reserves = gr.Number(label="Liquid Reserves ($)", value=120000.0)
            flags = gr.Textbox(label="Anomaly Flags (comma-separated)", placeholder="e.g. Missing tax transcript")
            
            submit_btn = gr.Button("Generate Risk Memo", variant="primary")
            
        with gr.Column(scale=2):
            gr.Markdown("### AI Underwriting Decision")
            with gr.Row():
                out_decision = gr.Textbox(label="Final Decision", show_label=True, scale=1)
                out_dti = gr.Textbox(label="Calculated DTI", scale=1)
                out_runway = gr.Textbox(label="Liquidity Runway", scale=1)
                
            with gr.Row():
                out_breaches = gr.Textbox(label="Policy Breaches", lines=3, scale=1)
                out_factors = gr.Textbox(label="Key Risk Factors", lines=3, scale=1)
                
            with gr.Accordion("Raw JSON Underwriting Memo", open=False):
                out_json = gr.JSON()

    submit_btn.click(
        fn=analyze_risk,
        inputs=[app_id, stated_inc, verified_inc, debt, reserves, flags],
        outputs=[out_decision, out_dti, out_runway, out_breaches, out_factors, out_json]
    )