"""
FastAPI web gateway for the automated underwriting pipeline.

Routes are thin: authentication/authorisation, validation and HTTP concerns live here, while all
decision logic lives in ``PipelineOrchestrator`` (src/engine/orchestrator.py).
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from starlette.middleware.base import RequestResponseEndpoint
from starlette.responses import Response

from src.api.auth import AuthenticatedUser, RoleChecker, UserRole
from src.api.middleware import RequestContextMiddleware, enforce_rate_limit
from src.common.config import settings
from src.common.logger import get_logger, request_id_ctx
from src.common.schemas import (
    DecisionPage,
    DecisionRecord,
    UnderwriteRequest,
    UnderwritingDecision,
)
from src.engine.decision_store import DecisionStore
from src.engine.orchestrator import PipelineOrchestrator

logger = get_logger("api")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    store = DecisionStore(settings.decision_store_path)
    store.open()
    app.state.store = store
    app.state.orchestrator = PipelineOrchestrator(store=store)
    logger.info(
        "startup",
        extra={"environment": settings.environment, "llm_review": settings.llm_review_enabled},
    )
    try:
        yield
    finally:
        # Graceful shutdown: uvicorn drains in-flight requests on SIGTERM, then we close the DB.
        store.close()
        logger.info("shutdown")


app = FastAPI(
    title="Aegis Underwriting API",
    description=(
        "Deterministic, explainable loan underwriting with a non-binding LLM advisory review. "
        "Authenticate with the `X-API-Key` header."
    ),
    version="1.1.0",
    lifespan=lifespan,
    docs_url="/docs" if settings.docs_enabled else None,
    redoc_url="/redoc" if settings.docs_enabled else None,
    openapi_url="/openapi.json" if settings.docs_enabled else None,
)


@app.middleware("http")
async def sanitize_unhandled_errors(request: Request, call_next: RequestResponseEndpoint) -> Response:
    """Global error boundary: logs the traceback, returns a generic body with the request id."""
    try:
        return await call_next(request)
    except Exception:
        logger.exception("unhandled_error", extra={"path": request.url.path})
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error", "request_id": request_id_ctx.get()},
        )


# --- Middleware (last added is outermost): CORS -> RequestContext -> error boundary -> routes ---
app.add_middleware(RequestContextMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["X-API-Key", "Content-Type", "X-Request-ID"],
    expose_headers=["X-Request-ID"],
)


# --- RBAC Role Guards ---
allow_underwriting = RoleChecker([UserRole.ADMIN, UserRole.UNDERWRITER])
allow_audit_read = RoleChecker([UserRole.ADMIN, UserRole.UNDERWRITER, UserRole.AUDITOR])


def get_orchestrator(request: Request) -> PipelineOrchestrator:
    return request.app.state.orchestrator


def get_store(request: Request) -> DecisionStore:
    return request.app.state.store


# --- Probes ---
@app.get("/health", tags=["ops"])
async def health_check() -> dict:
    """Public liveness probe for Modal and uptime monitors."""
    return {"status": "ok", "environment": settings.environment}


@app.get("/ready", tags=["ops"])
async def readiness_check(store: DecisionStore = Depends(get_store)) -> JSONResponse:
    """Readiness: the decision ledger is reachable. Does not wake the GPU worker."""
    try:
        await store.aping()
    except Exception:
        logger.exception("readiness_failed")
        return JSONResponse(status_code=503, content={"status": "unavailable"})
    return JSONResponse(content={"status": "ready", "llm_review": settings.llm_review_enabled})


@app.get("/auth/me", response_model=AuthenticatedUser, tags=["auth"])
async def read_current_user(
    current_user: AuthenticatedUser = Depends(allow_audit_read),
) -> AuthenticatedUser:
    """Returns the authenticated caller's identity and assigned RBAC role."""
    return current_user


# --- Underwriting ---
@app.post(
    "/underwrite",
    response_model=DecisionRecord,
    tags=["underwriting"],
    dependencies=[Depends(enforce_rate_limit)],
)
async def process_underwriting(
    request: UnderwriteRequest,
    current_user: AuthenticatedUser = Depends(allow_underwriting),
    orchestrator: PipelineOrchestrator = Depends(get_orchestrator),
) -> DecisionRecord:
    """Underwrites an application and returns the persisted, explainable decision."""
    return await orchestrator.underwrite(
        request, decided_by=current_user.user_id, decided_by_role=current_user.role.value
    )


@app.post(
    "/underwrite/document",
    response_model=DecisionRecord,
    tags=["underwriting"],
    dependencies=[Depends(enforce_rate_limit)],
)
async def process_underwriting_document(
    application: str = Form(..., description="UnderwriteRequest as a JSON string"),
    file: UploadFile = File(..., description="Supporting financial PDF"),
    current_user: AuthenticatedUser = Depends(allow_underwriting),
    orchestrator: PipelineOrchestrator = Depends(get_orchestrator),
) -> DecisionRecord:
    """Underwrites an application with a supporting PDF (bank statement, financials)."""
    try:
        request = UnderwriteRequest.model_validate_json(application)
    except ValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail=exc.errors(include_url=False, include_context=False, include_input=False),
        ) from None

    max_bytes = int(settings.max_upload_mb * 1024 * 1024)
    content = await file.read(max_bytes + 1)
    if len(content) > max_bytes:
        raise HTTPException(status_code=413, detail=f"File exceeds {settings.max_upload_mb} MB.")
    if not content.startswith(b"%PDF-"):
        raise HTTPException(status_code=415, detail="Only PDF documents are accepted.")

    filename = (file.filename or "document.pdf")[:255]
    return await orchestrator.process_application(
        request,
        content,
        filename,
        decided_by=current_user.user_id,
        decided_by_role=current_user.role.value,
    )


# --- Decision ledger (audit) ---
@app.get("/decisions", response_model=DecisionPage, tags=["decisions"])
async def list_decisions(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    application_id: str | None = Query(None, max_length=64),
    decision: UnderwritingDecision | None = None,
    _: AuthenticatedUser = Depends(allow_audit_read),
    store: DecisionStore = Depends(get_store),
) -> DecisionPage:
    """Paginated decision history, newest first. Readable by auditors."""
    return await store.list(
        limit=limit, offset=offset, application_id=application_id, decision=decision
    )


@app.get("/decisions/{decision_id}", response_model=DecisionRecord, tags=["decisions"])
async def get_decision(
    decision_id: str,
    _: AuthenticatedUser = Depends(allow_audit_read),
    store: DecisionStore = Depends(get_store),
) -> DecisionRecord:
    record = await store.get(decision_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Decision not found.")
    return record
