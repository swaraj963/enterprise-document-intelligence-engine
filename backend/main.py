import json
import logging
import re
import secrets
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from starlette.concurrency import run_in_threadpool

from .config import ROOT, get_settings
from .rag_engine import RAGEngine
from .security import (
    SecurityHeadersMiddleware, TenantMiddleware, UnsafePrompt, UploadRejected, limiter,
    safe_storage_name, sanitize_filename, sanitize_question, validate_upload,
)

settings = get_settings()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("engine")
engine: RAGEngine | None = None

SESSION_SECRET = settings.session_secret.get_secret_value()
if not SESSION_SECRET:
    SESSION_SECRET = secrets.token_hex(32)
    log.warning("SESSION_SECRET not set: using a temporary one. Visitors lose access to "
                "their vaults on restart. Set SESSION_SECRET in .env.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    log.info("Loading knowledge engine (first run downloads a small model)...")
    engine = RAGEngine(settings)
    if not engine.llm_ready:
        log.warning("GROQ_API_KEY is not set. Uploads work, but answering is disabled.")
    yield


app = FastAPI(
    title="Enterprise Document Intelligence Engine",
    lifespan=lifespan,
    docs_url="/api/docs" if settings.debug else None,
    redoc_url=None,
    openapi_url="/api/openapi.json" if settings.debug else None,
)

# Middleware order: last added runs first.
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)
app.add_middleware(TenantMiddleware, secret=SESSION_SECRET, secure=settings.cookie_secure)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.origins,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type"],
    allow_credentials=False,
    max_age=600,
)

DOC_ID_RE = re.compile(r"^[0-9a-f]{32}$")


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=settings.max_question_chars * 2)


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):
    log.exception("Unhandled error")
    return JSONResponse({"detail": "Something went wrong. Please try again."}, status_code=500)


@app.get("/api/health")
def health():
    return {"status": "ok", "ai_ready": bool(engine and engine.llm_ready)}


@app.get("/api/documents")
@limiter.limit(settings.rate_limit_default)
def list_documents(request: Request):
    return {"documents": engine.list_documents(request.state.tenant)}


@app.post("/api/documents", status_code=201)
@limiter.limit(settings.rate_limit_upload)
async def upload_document(request: Request, file: UploadFile = File(...)):
    tenant = request.state.tenant
    max_bytes = settings.max_upload_mb * 1024 * 1024
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > max_bytes + 4096:
        raise HTTPException(413, f"Files must be under {settings.max_upload_mb} MB.")

    buf = bytearray()
    while chunk := await file.read(1024 * 256):
        buf.extend(chunk)
        if len(buf) > max_bytes:
            raise HTTPException(413, f"Files must be under {settings.max_upload_mb} MB.")
    data = bytes(buf)

    display_name = sanitize_filename(file.filename)
    try:
        ext = validate_upload(display_name, data)
    except UploadRejected as e:
        raise HTTPException(415, str(e))

    tmp = settings.upload_tmp_dir / safe_storage_name(ext)
    doc_id = uuid4().hex
    try:
        tmp.write_bytes(data)
        count = await run_in_threadpool(
            engine.ingest, tenant, tmp, display_name, ext, len(data), doc_id
        )
    except ValueError as e:
        raise HTTPException(422, str(e))
    except Exception:
        log.exception("Ingestion failed")
        raise HTTPException(422, "We couldn't read this file. It may be damaged.")
    finally:
        tmp.unlink(missing_ok=True)

    return {"id": doc_id, "name": display_name, "sections": count}


@app.delete("/api/documents/{doc_id}")
@limiter.limit(settings.rate_limit_default)
def delete_document(request: Request, doc_id: str):
    if not DOC_ID_RE.match(doc_id):
        raise HTTPException(400, "Invalid document id.")
    engine.delete_document(request.state.tenant, doc_id)
    return {"deleted": doc_id}


def _prepare(body: ChatRequest) -> str:
    if not engine.llm_ready:
        raise HTTPException(503, "The AI service isn't configured yet. Ask your administrator to add an API key.")
    try:
        return sanitize_question(body.question, settings.max_question_chars)
    except UnsafePrompt as e:
        raise HTTPException(400, str(e))


@app.post("/api/chat")
@limiter.limit(settings.rate_limit_chat)
def chat(request: Request, body: ChatRequest):
    question = _prepare(body)
    try:
        return engine.ask(question, request.state.tenant)
    except Exception:
        log.exception("Query failed")
        raise HTTPException(502, "The AI service is busy right now. Please try again in a moment.")


@app.post("/api/chat/stream")
@limiter.limit(settings.rate_limit_chat)
def chat_stream(request: Request, body: ChatRequest):
    question = _prepare(body)  # validation errors return normal JSON before streaming starts
    tenant = request.state.tenant

    def events():
        try:
            for ev in engine.ask_stream(question, tenant):
                yield f"data: {json.dumps(ev)}\n\n"
        except Exception:
            log.exception("Streaming query failed")
            err = {"type": "error", "message": "The AI service is busy right now. Please try again in a moment."}
            yield f"data: {json.dumps(err)}\n\n"

    return StreamingResponse(
        events(), media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-store"},
    )


# Static frontend (mounted last so /api/* wins)
app.mount("/", StaticFiles(directory=ROOT / "frontend", html=True), name="ui")