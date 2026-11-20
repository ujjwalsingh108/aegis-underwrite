# Fixes Applied

See `PRODUCTION_READINESS_REPORT.md` for the findings behind each change.

## Files

### Created
| File | Purpose |
|---|---|
| `src/common/prompts.py` | Canonical ChatML prompt shared by training and serving |
| `src/engine/verification.py` | Verified-metrics aggregation + income variance (moved from the parser) |
| `src/engine/decision_store.py` | SQLite decision ledger (async facade, indexed, one transaction per decision) |
| `src/api/middleware.py` | Request ID, security headers, access log, rate limiter |
| `src/ui/__init__.py` | Package marker |
| `.env.example` | Every supported variable, documented |
| `tests/test_config.py` | Startup security validation tests |
| `PRODUCTION_READINESS_REPORT.md`, `FIXES_APPLIED.md` | Reports |

### Deleted
| File | Justification |
|---|---|
| `src/api/dependencies.py` | Duplicate, weaker auth (master key only, non-constant-time `!=`). Grep confirmed 0 importers. |

### Modified
| File | Change |
|---|---|
| `src/common/config.py` | `SecretStr` keys; env normalisation; startup security validation; new settings (store path, LLM toggle/timeout, rate limit, upload cap, docs, log level, Groq, policy ratios); no longer loads `.env.example` as config |
| `src/common/logger.py` | Was empty: JSON logger + request-id context |
| `src/common/schemas.py` | Your uncommitted memo validator kept. Added numeric bounds, `DECISION_SEVERITY`, `UnderwriteRequest`, `DecisionRecord`, `DecisionPage`, `LLMAdvisory`, `UnderwritingTrainingSample`; infinity-safe JSON |
| `src/engine/pipeline.py` | Rule IDs, reasons, `evaluate()` → `PipelineResult`; uses ratio-engine LTV; corporate-without-financials → MANUAL_REVIEW; thresholds from settings |
| `src/engine/ratio_calculator.py` | Conservative qualifying figures when verification data is present; `missing_financials` flag |
| `src/engine/guardrails.py` | Was empty: LLM advisory reconciliation (escalate-only) |
| `src/engine/orchestrator.py` | Rewritten as the single underwriting service (JSON + document); hardcoded `APP-9921` removed |
| `src/api/main.py` | Thin routes over the orchestrator; lifespan; error boundary; `/ready`, `/underwrite/document`, `/decisions`, `/decisions/{id}`; tighter CORS; docs toggle |
| `src/api/auth.py` | `SecretStr`, bytes-safe constant-time comparison across all keys |
| `src/vision/parser.py` | PDF extraction only; mock verification defaults removed |
| `src/serving/vllm_engine.py` | Version-tolerant structured outputs (fixes vLLM 0.4.2 crash); logging; no raw output in exceptions; canonical prompt in smoke entrypoint |
| `src/serving/modal_app.py` | Gateway image gets `pypdf`/`pdfplumber`; unneeded volume mount and `requests` removed |
| `src/training/dataset_generator.py` | Canonical prompt, row validation, settings instead of `os.environ`/`load_dotenv`, logging |
| `src/training/lora_trainer.py` | Uses shared `to_chatml`; resume only if a checkpoint exists; `torch.load` bypass scoped to resume and documented |
| `src/training/modal_train.py` | Logging |
| `src/ui/dashboard.py` | API URL/key from env, sends `X-API-Key`, new request/response contract, correct DTI formatting, `launch()` entrypoint |
| `pyproject.toml` | Lean base deps + `inference`/`train`/`data`/`ui`/`dev` extras; unused deps removed (docling, scikit-learn, pandas, numpy, pillow, pyyaml, rich, trl, bitsandbytes); `python-multipart` added; ruff FastAPI config; mypy config |
| `.gitignore` | `!.env.example`; ignore local `data/*.db` |
| `README.md` | Run, test, deploy instructions; endpoint table; decision flow |
| `tests/*` | Rewritten `test_api.py` / `test_orchestrator.py` (were stale); new guardrail, ratio, parser, prompt and config tests; xfail removed (bug already fixed); conftest pins test env |

## Before/after: major deduplication

**Decision assembly** existed twice (`api/main.py` and `engine/orchestrator.py`), each doing *LLM decides → override APPROVE to MANUAL_REVIEW if ratio breaches*:

```python
# before (both files)
memo = await engine.generate_memo.remote.aio(prompt)          # LLM is the decision
merged_breaches = list(dict.fromkeys([*ratios["breached_policies"], *memo.breached_policies]))
final_decision = MANUAL_REVIEW if memo.decision == APPROVE and merged_breaches else memo.decision
```

```python
# after (src/engine/orchestrator.py, used by both routes)
result = self.pipeline.evaluate(profile, metrics)                         # rules engine decides
llm_memo, status, detail = await self._llm_advisory(profile, metrics, excerpt)
final_memo, advisory = reconcile_with_advisory(result.memo, llm_memo, status, detail)  # escalate-only
await self.store.save(record); audit_logger.info("underwriting_decision", extra={...})
```

**Prompts**: four hand-written variants → `src/common/prompts.py`. Training rows and inference prompts now differ only by the completion (enforced by `tests/test_training.py::test_training_and_inference_prompts_share_a_prefix`).

## API contract changes (breaking)

`POST /underwrite` request now **requires** `requested_loan_amount` (> 0), `kyc_confidence_score` and `document_tampering_score` (0–1). New optional fields: `entity_type`, `detected_monthly_liabilities`, `verified_liquid_reserves`, collateral, corporate financials, `synthetic_voice_probability`, `pep_or_sanctions_hit`, `ubo_verified`.

The response is now a `DecisionRecord`:
`{decision_id, request_id, decided_at, decided_by, decided_by_role, decision_rule, decision_reasons, memo: UnderwritingMemo, llm_advisory, source_document}`. Infinite ratios appear as `"Infinity"`.

## How to run now

```bash
pip install -e ".[dev]"
cp .env.example .env          # set MASTER_API_KEY, UNDERWRITER_API_KEY, AUDITOR_API_KEY (distinct, >=16 chars)

pytest                                     # 54 tests
ruff check src tests
mypy src --exclude src/ui

LLM_REVIEW_ENABLED=false uvicorn src.api.main:app --port 8000   # local, no GPU
curl -s localhost:8000/health

curl -s -X POST localhost:8000/underwrite -H "X-API-Key: $UNDERWRITER_API_KEY" \
  -H "content-type: application/json" -d '{"applicant_id":"APP-1","requested_loan_amount":250000,
  "stated_monthly_income":12000,"verified_monthly_income":11800,"total_monthly_debt":3500,
  "liquid_reserves":60000,"kyc_confidence_score":0.97,"document_tampering_score":0.02}'
```

Required env vars: `MASTER_API_KEY`, `UNDERWRITER_API_KEY`, `AUDITOR_API_KEY` (strictly required in staging/production). Everything else has a safe default; see `.env.example`. On Modal, add any overrides (for example `ENVIRONMENT=production`, `DECISION_STORE_PATH`) to the `aegis-rbac-keys` secret.
