# Aegis-Underwrite: Production Readiness Report

Audit date: 2026-11-20 · Scope: every file in the repository (21 source modules, 9 test modules, config, dataset)

## Verdict

**Before this work, the repository would not have worked in production.** The gateway booted and `/health` returned 200. But on the deployed GPU image every `/underwrite` call failed: the vLLM worker imported an API that does not exist in the pinned `vllm==0.4.2`. Even with that fixed, the binding decision came from the LLM. It was prompted in a format different from the one it was trained on, and the real rules engine was never called at serve time.

**After this work**, the gateway runs end to end locally. A deterministic, explainable, persisted decision is produced for every application. The full test suite, lint and typecheck pass. The service is **ready for a staging deployment**. It is **not yet production-ready** until the items in "Remaining manual decisions" are resolved, chiefly a durable managed database and a GPU-path verification run on Modal.

> What is this system? It is a **loan/credit underwriting engine** (DTI, DSCR, LTV, liquidity, KYC/fraud, sanctions), not an insurance policy system. There is no premium/rating engine and no "list policies" concept. "Policy" here means credit policy. Domain checklist items for insurance pricing are marked N/A below rather than invented.

---

## 1. Initial state: what was broken

### Build & runtime health (before)

| Check | Command | Result | Evidence / file |
|---|---|---|---|
| Install | `.venv` present, Python 3.11.9 | PASS (with conflicts) | `pip check`: numpy 1.26.4 vs pyarrow/opencv needing ≥2; protobuf 4.25 vs vllm/opentelemetry |
| Compile | `python -m compileall src tests` | PASS | n/a |
| Tests | `pytest` | **FAIL**: 1 collection error, 2 failures, 1 unexpected pass | `tests/test_training.py` could not import (`lora_trainer.py` pulled in `datasets`→`pyarrow`→numpy conflict); `test_api.py` and `test_orchestrator.py` were stale vs code (`.remote` vs `.remote.aio`); xfail in `test_ratio_calculator.py` was already fixed |
| Lint | `ruff check src tests` | **FAIL**: 101 issues | unused imports, unsorted imports, redefinitions, `raise` without `from` |
| Typecheck | `mypy src` | not configured | n/a |
| Start | `uvicorn src.api.main:app` | PASS | `/health` → 200, bad key → 401 |
| Underwrite on Modal | n/a (static analysis) | **FAIL** | `src/serving/vllm_engine.py` imported `GuidedDecodingParams` and passed `guided_decoding=` to `SamplingParams`. Neither exists in vLLM 0.4.2, which `src/serving/modal_app.py` pins (verified against the v0.4.2 source). Every `generate_memo` call raised `ImportError`. |
| DB / migrations | n/a | **FAIL**: none | No persistence: decisions could not be retrieved or audited |
| Env documentation | n/a | **FAIL** | No `.env.example`, and `.gitignore` rule `.env.*` would have ignored one. `config.py` also loaded `.env.example` as a live config fallback. |

### Critical defects found

| # | Defect | Location | Impact |
|---|---|---|---|
| C1 | vLLM API mismatch (above) | `src/serving/vllm_engine.py` | All LLM inference fails on deploy |
| C2 | LLM made the binding decision. The deterministic rules engine (`UnderwritingPipeline`), which also generates the training labels, was never called at serve time. Sanctions/fraud rules were not enforced on `/underwrite`, and an LLM `REJECT_*` was passed through as an unexplained adverse action. | `src/api/main.py` | Non-deterministic, unexplainable decisions |
| C3 | Train/serve prompt skew: the LoRA was trained on the `dataset_generator` format but served three different hand-built prompts | `api/main.py`, `engine/orchestrator.py`, `serving/vllm_engine.py` vs `training/dataset_generator.py` | Adapter sees an out-of-distribution input |
| C4 | Orchestrator ignored the uploaded document and returned a hardcoded applicant `APP-9921` for every PDF, and it was not wired to any route | `src/engine/orchestrator.py:37-46` | Dead/mock path |
| C5 | Placeholder API keys (`"MASTER_API_KEY"` etc.) accepted in every environment; duplicate keys silently collapsed roles in the key registry | `src/common/config.py`, `src/api/auth.py` | Privilege escalation |
| C6 | `500` responses returned `str(e)`; LLM validation errors returned the raw model output (applicant PII) to the client | `src/api/main.py:152-157`, `vllm_engine.py:135` | Internal and PII leakage |
| C7 | Infinite ratios (zero debt / zero income) serialized as `null`, so a stored memo with infinite runway could not be read back | `src/common/schemas.py` | Data loss / 500s |
| C8 | Corporate applicant with no financials got `CONDITIONAL_APPROVAL` for the full amount | `src/engine/pipeline.py` waterfall | Credit policy gap |
| C9 | Ratios computed on *stated* claims; verified figures only affected one income check | `src/engine/ratio_calculator.py` | Applicant-supplied numbers drive DTI/runway |
| C10 | KYC/tampering defaulted to `0.95` / `0.01` when not supplied | `src/vision/parser.py:55-56` | Fabricated verification results |
| C11 | `trainer.train(resume_from_checkpoint=True)` raises on a fresh run (no checkpoint) | `src/training/lora_trainer.py:173` | First training run crashes |
| C12 | Dashboard sent no `X-API-Key` (every call 401), hardcoded a personal Modal URL, showed DTI `0.35` as `0.35%`, and crashed on `null` DTI | `src/ui/dashboard.py` | UI non-functional |

## 2. Completeness audit

| File | Line (before) | Incomplete logic | Severity | Resolution |
|---|---|---|---|---|
| `src/engine/orchestrator.py` | 21, 36-46 | Vision agent commented out; hardcoded `parsed_data` placeholder | High | Placeholder removed: figures come from the request, and the PDF text goes to the advisor. **Vision extraction itself remains unimplemented** (manual decision M3). |
| `src/engine/guardrails.py` | n/a | Empty file | Medium | Implemented LLM-advisory reconciliation guardrail |
| `src/common/logger.py` | n/a | Empty file | Medium | Implemented structured JSON logger with request-id correlation |
| `src/vision/__init__.py` | 1 | Milestone note only | Low | Cleaned |
| `src/api/main.py` | n/a | No decision retrieval/listing, no document upload route, `allow_admin_only` unused | High | Added `/underwrite/document`, `/decisions`, `/decisions/{id}`, `/ready`. Removed unused guard. |
| `src/vision/parser.py` | 46-57 | Mock default verification scores | High | Scores now required (C10) |
| Audit logging | n/a | None | High | Ledger + `underwriting_decision` audit log line per decision |
| Risk scoring | n/a | No `random()`/stub; real ratio + waterfall engine exists | n/a | Kept. No composite credit score exists by design (M5). |
| Premium / rate engine | n/a | Not part of this product (loan underwriting) | N/A | M6 |

Grep for `TODO|FIXME|XXX|HACK|NotImplemented|APP-9921` in `src/` after the fixes: **0 matches**.

## 3. Duplication report

| Logic | Locations (before) | Kept | Consolidated into |
|---|---|---|---|
| Ratio-merge + decision override + memo assembly | `api/main.py:75-151` and `engine/orchestrator.py:59-119` (near-verbatim copies) | Neither; both replaced | `PipelineOrchestrator.underwrite()` + `engine/guardrails.reconcile_with_advisory()` |
| Prompt construction (4 variants) | `api/main.py:94-110`, `orchestrator.py:63-78`, `vllm_engine.py:153-164`, `dataset_generator.py:62-76` + `lora_trainer.format_prompt` | Training format (what the adapter learned) | `src/common/prompts.py` (`SYSTEM_INSTRUCTION`, `build_input_context`, `to_chatml`, `build_inference_prompt`) |
| Income variance % | `api/main.py:76-81`, `vision/parser.py:63-68` | parser version (rounded) | `engine/verification.income_variance_pct` |
| LTV | `engine/pipeline.py:43-45` (unrounded, no zero guard), `ratio_calculator.calculate_ltv` | `ratio_calculator` | Pipeline now uses the ratio engine's value |
| API-key auth | `api/dependencies.py` (master key only, `!=` comparison) and `api/auth.py` (RBAC, constant-time) | `auth.py` | `dependencies.py` deleted (0 importers) |
| Fraud-probability aggregation | inline in pipeline | n/a | `pipeline.overall_fraud_probability()` |
| Thresholds `0.75`, `0.85`, `0.15` | hardcoded in `pipeline.py`, `parser.py` | n/a | `settings.severe_income_shortfall_ratio`, `income_mismatch_flag_ratio`, `tampering_flag_threshold` |
| `UnderwritingTrainingSample` | in `lora_trainer.py` (with a stray `@dataclass` on a `BaseModel`), never used by the trainer | n/a | `src/common/schemas.py`, now used by `dataset_generator` to validate every row |

There were no `engine_v2` or `old_engine` folders. The duplication was copy-paste between the API and the orchestrator.

## 4. Connections fixed

| From | To | Issue | Fix |
|---|---|---|---|
| `POST /underwrite` | `UnderwritingPipeline` | Rules engine not called | Route → `PipelineOrchestrator` → pipeline → guardrail → store |
| (no route) | `PipelineOrchestrator.process_application` | Orchestrator unreachable | `POST /underwrite/document` |
| API | persistence | None existed | `DecisionStore`, opened/closed in FastAPI lifespan, injected via `Depends` |
| `modal_app.gateway_image` | `vision/parser.py` deps | `pypdf`/`pdfplumber` missing from gateway image | Added (the document route would have `ImportError`ed on deploy) |
| `modal_app.web_endpoint` | Volume | Mounted model volume on a CPU gateway that never reads it | Removed mount |
| Serving prompt | Training prompt | Skew (C3) | Shared `src/common/prompts.py` |
| `dataset_generator` | config | Read `os.environ` + `load_dotenv` directly | `settings.groq_api_key` / `settings.groq_model` |
| `dashboard.py` | API auth | No API key header | `AEGIS_API_KEY` header + new response envelope |
| `pyproject.toml` | runtime | `python-multipart` (needed for uploads) undeclared | Declared |

## 5. Hardening applied

| Area | What was added | Where |
|---|---|---|
| Config | Single validated settings object; env aliases; `SecretStr` keys; startup failure on placeholder/short/duplicate keys outside dev; `*` CORS rejected outside dev; bounded thresholds; `extra="ignore"` | `src/common/config.py` |
| Validation | Bounds on every numeric input (`ge`/`gt`/`le`), ID pattern, flag count/length limits, required KYC/tampering scores, PDF magic-byte + size checks | `src/common/schemas.py`, `src/api/main.py` |
| Error handling | Global error boundary returning `{"detail","request_id"}` with server-side traceback; LLM errors no longer carry raw output | `src/api/main.py`, `src/serving/vllm_engine.py` |
| Logging | JSON logs, request-id correlation, access log, per-decision audit event; all `print` removed | `src/common/logger.py`, `src/api/middleware.py` |
| Security headers | nosniff, frame DENY, no-referrer, CORP, HSTS, `Cache-Control: no-store` (Helmet-equivalent) | `src/api/middleware.py` |
| CORS | Explicit methods/headers, `X-Request-ID` exposed | `src/api/main.py` |
| Rate limiting | Sliding window per API identity on underwriting routes (429 + `Retry-After`) | `src/api/middleware.py` |
| Auth | Constant-time comparison over all keys, bytes-safe (non-ASCII keys no longer 500) | `src/api/auth.py` |
| Probes | `/health` (liveness), `/ready` (ledger reachable; does not wake the GPU) | `src/api/main.py` |
| Shutdown | Lifespan closes the store after uvicorn drains requests; vLLM `@modal.exit` retained | `src/api/main.py` |
| OpenAPI | Tagged routes, typed responses; docs off by default in production | `src/api/main.py` |
| Explainability | `decision_rule` (stable IDs R1–R7, G1), `decision_reasons`, audit trail entries, `llm_advisory` block | `src/engine/pipeline.py`, `guardrails.py` |
| Resilience | LLM call has a timeout; LLM failure never fails the request | `src/engine/orchestrator.py` |
| vLLM compatibility | Structured-output kwargs resolved for any vLLM version (`StructuredOutputsParams` → `GuidedDecodingParams` → Pydantic-only) | `src/serving/vllm_engine.py` |
| Non-blocking I/O | SQLite and PDF parsing run in worker threads | `decision_store.py`, `orchestrator.py` |

## 6. Final verification (after)

| Check | Command | Result |
|---|---|---|
| Compile | `python -m compileall -q src tests` | PASS |
| Lint | `ruff check src tests` | PASS: "All checks passed!" |
| Typecheck | `mypy src --exclude src/ui` | PASS: "no issues found in 26 source files" |
| Tests | `pytest --cov=src` | PASS: **54 passed**, 73% coverage (uncovered: Modal/GPU/Gradio modules) |
| Start | `uvicorn src.api.main:app` (ENVIRONMENT=staging, LLM off) | PASS |
| Existing `.env` | load settings with new validation | PASS (values not printed) |

Live smoke test against the running server:

```
GET  /health  -> {"status":"ok","environment":"staging"}
GET  /ready   -> {"status":"ready","llm_review":false}
POST /underwrite (individual, clean)
     -> APPROVE  rule=R7_ALL_POLICIES_PASSED  dti=0.2966  runway=17.14  fraud=0.03  approved=250000.0
POST /underwrite (LLC, sanctions hit)
     -> REJECT_FRAUD  rule=R1_SANCTIONS_HIT  approved=0.0
        reasons=['Applicant or beneficial owner matched a sanctions/PEP watchlist.']
GET  /decisions/{id} (auditor)        -> 200
GET  /decisions?limit=10 (auditor)    -> total 2 ['REJECT_FRAUD', 'APPROVE']
POST /underwrite as auditor           -> 403
POST /underwrite invalid payload      -> 422
headers: x-request-id, x-content-type-options: nosniff, x-frame-options: DENY, cache-control: no-store
audit log: {"logger":"aegis.audit","msg":"underwriting_decision","decision_id":"bad273ab-…",
            "applicant_id":"APP-SMOKE-1","decision":"APPROVE","decision_rule":"R7_ALL_POLICIES_PASSED",…}
```

**Not verified here:** `modal deploy` / `modal run` (needs Modal credentials and a GPU), the LoRA training job, the Gradio UI, and graceful shutdown on SIGTERM (Windows `kill` hard-terminates; lifespan shutdown is exercised by the test client).

## 7. Remaining manual decisions

| # | Decision | Why it needs an owner |
|---|---|---|
| M1 | **Durable decision database.** SQLite is fine locally, but on Modal the gateway filesystem is ephemeral and SQLite must not be shared across containers via a Volume. Pick Postgres (or Modal Dict for low volume) and implement it behind `DecisionStore`. Until then the JSON audit log is the durable record. | Infra/compliance; record-retention rules |
| M2 | **Verify the GPU path on Modal** (`modal run src/serving/vllm_engine.py`). Consider upgrading the GPU image (vLLM 0.4.2 / torch 2.3) to get constrained JSON decoding and drop the `torch.load` safety bypass in `lora_trainer.py`. | Needs GPU and credentials |
| M3 | **Document → figures extraction** (Qwen2-VL vision agent) is not implemented. `/underwrite/document` currently uses request figures and passes the PDF text only to the advisor. | Model/product scope |
| M4 | **API contract change.** `/underwrite` now requires `requested_loan_amount`, `kyc_confidence_score` and `document_tampering_score`, and returns a `DecisionRecord` envelope (memo under `memo`). The bundled dashboard is updated; any external callers must migrate. | Client coordination |
| M5 | **Credit-policy sign-off** on behaviour introduced or made explicit: conservative qualifying figures; corporate-without-financials → MANUAL_REVIEW; LLM may only escalate to MANUAL_REVIEW; LLM failure keeps the deterministic decision. Whether to add a composite numeric risk score (none exists; inventing weights is a policy call). Whether CONDITIONAL with an LTV breach should cap the amount at `collateral × max LTV`. Whether MANUAL_REVIEW should show `approved_loan_amount=0`. | Credit risk |
| M6 | Premium/pricing engine: not part of this product. Confirm it is out of scope. | Product |
| M7 | Rate limits are per container. For a hard global cap use a shared store or Modal concurrency limits. | Infra |
| M8 | Training quality: loss is computed over prompt and padding tokens (no assistant-only masking), and training labels with infinite ratios were serialized as `null`. Regenerating the dataset with the current pipeline fixes the second. | ML |
| M9 | Local venv has numpy 1.26.4, which breaks `pyarrow`/`datasets` imports (`pip check`). Run `pip install "numpy>=2"` after confirming nothing else pins numpy<2. | Environment owner |
| M10 | Replace shared API keys with per-user identities (JWT via `auth_jwt_issuer`) so the audit trail names a person, not `underwriter-svc`. | Security/compliance |

## 8. Production checklist: final status

| Category | Check | Status | File |
|---|---|---|---|
| Error handling | Global handler, no stack-trace leak | PASS | `src/api/main.py` |
| Validation | Pydantic bounds on all inputs | PASS | `src/common/schemas.py` |
| Security | No hardcoded secrets, insecure keys rejected, RBAC on all non-probe routes, CORS, security headers, rate limit | PASS (rate limit per container, M7) | `config.py`, `auth.py`, `middleware.py` |
| Explainability | Rule ID + reasons + audit trail on every decision | PASS | `pipeline.py`, `guardrails.py` |
| Logging | Structured JSON, request ID, audit event per decision | PASS | `logger.py`, `orchestrator.py` |
| Config | Central validated config, no scattered env reads (UI client excepted) | PASS | `config.py` |
| DB | Transaction per decision, indexes, schema auto-created | PASS locally / **FAIL for prod durability** (M1) | `decision_store.py` |
| API | Pagination, consistent envelope, OpenAPI | PASS | `main.py` |
| Testing | Tests exist and pass | PASS (54) | `tests/` |
| Performance | No blocking I/O on the event loop; LLM timeout | PASS | `orchestrator.py`, `decision_store.py` |
| GPU inference on Modal | Verified end to end | **NOT VERIFIED** (M2) | `vllm_engine.py` |
