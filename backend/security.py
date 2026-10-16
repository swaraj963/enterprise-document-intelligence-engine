"""Upload validation, prompt-injection guardrails, rate limiting, security headers,
and private-vault session cookies."""
import hashlib
import hmac
import io
import re
import unicodedata
import uuid
import zipfile
from pathlib import PurePath

from slowapi import Limiter
from slowapi.util import get_remote_address
from starlette.middleware.base import BaseHTTPMiddleware

# ------------------------------------------------------------------ rate limiting
limiter = Limiter(key_func=get_remote_address, default_limits=["60/minute"])

# ------------------------------------------------------------------ files
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".csv", ".json", ".log"}


class UploadRejected(ValueError):
    """Raised with a user-safe message when an upload fails validation."""


def sanitize_filename(raw):
    """Return a safe display name: no directories, no control chars, bounded length."""
    name = (raw or "document").replace("\\", "/")
    name = PurePath(name).name  # strips any path components (../../etc/passwd)
    name = unicodedata.normalize("NFKC", name)
    name = re.sub(r"[\x00-\x1f\x7f]", "", name)
    name = re.sub(r"[^\w\s.\-()]", "_", name, flags=re.UNICODE).strip(" .")
    stem, dot, ext = name.rpartition(".")
    if not dot:
        stem, ext = name, ""
    stem = stem[:80] or "document"
    return f"{stem}.{ext.lower()}" if ext else stem


def safe_storage_name(ext: str) -> str:
    """Server-generated name: user input never touches the filesystem path."""
    return f"{uuid.uuid4().hex}{ext}"


def validate_upload(filename: str, data: bytes) -> str:
    """Verify extension AND magic bytes. Returns normalized extension."""
    ext = PurePath(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise UploadRejected("Only PDF, DOCX and TXT files are supported.")
    if not data:
        raise UploadRejected("This file is empty.")

    if ext == ".pdf":
        if not data[:1024].lstrip().startswith(b"%PDF-"):
            raise UploadRejected("This file doesn't look like a real PDF.")
    elif ext == ".docx":
        if not data.startswith(b"PK\x03\x04"):
            raise UploadRejected("This file doesn't look like a real Word document.")
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                names = set(z.namelist())
                if "word/document.xml" not in names or "[Content_Types].xml" not in names:
                    raise UploadRejected("This file doesn't look like a real Word document.")
                if sum(i.file_size for i in z.infolist()) > 200 * 1024 * 1024:
                    raise UploadRejected("This document is too large once opened.")
        except zipfile.BadZipFile:
            raise UploadRejected("This Word document appears to be corrupted.")
    else:  # .txt
        if b"\x00" in data[:8192]:
            raise UploadRejected("This doesn't look like a plain text file.")
        try:
            data.decode("utf-8")
        except UnicodeDecodeError:
            raise UploadRejected("Text files must be UTF-8 encoded.")
    return ext


# ------------------------------------------------------------------ prompt safety
_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+|any\s+)?(the\s+)?(previous|prior|above|earlier|preceding)\s+(instructions?|prompts?|rules?|context)",
    r"disregard\s+(all\s+|any\s+)?(the\s+)?(previous|prior|above|your|system)\s+\w*",
    r"forget\s+(everything|all|your)\s+(you|instructions?|rules?|above)",
    r"(reveal|show|print|repeat|output|leak)\s+(me\s+)?(your|the)\s+(system\s+|hidden\s+|initial\s+)?(prompt|instructions?|rules?)",
    r"what\s+(is|are)\s+your\s+(system\s+)?(prompt|instructions?)",
    r"you\s+are\s+now\s+(a|an|in|the)\b",
    r"\b(act|behave|respond)\s+as\s+(if\s+you\s+(are|were)|though\s+you)\b",
    r"\bpretend\s+(to\s+be|you\s+are|that\s+you)\b",
    r"\b(jailbreak|jailbroken|DAN\s+mode|developer\s+mode|god\s+mode)\b",
    r"\bdo\s+anything\s+now\b",
    r"override\s+(your\s+)?(safety|rules|instructions|guidelines|restrictions)",
    r"<\|?(im_start|im_end|system|assistant|endoftext)\|?>",
    r"(^|\n)\s*(system|assistant)\s*:",
    r"\[/?INST\]|<<SYS>>|###\s*(system|instruction)",
    r"without\s+(any\s+)?(restrictions|filters|limitations|rules)",
]
_INJECTION_RE = re.compile("|".join(f"(?:{p})" for p in _INJECTION_PATTERNS), re.I)

_CONTROL_CHARS = r"[\x00-\x08\x0b-\x1f\x7f\u200b-\u200f\u202a-\u202e\u2060\ufeff]"


class UnsafePrompt(ValueError):
    pass


def sanitize_question(q: str, max_chars: int) -> str:
    q = unicodedata.normalize("NFKC", q)
    q = re.sub(_CONTROL_CHARS, "", q)
    q = re.sub(r"\s+", " ", q).strip()
    if len(q) < 2:
        raise UnsafePrompt("Please type a question.")
    if len(q) > max_chars:
        raise UnsafePrompt(f"Please keep questions under {max_chars} characters.")
    if _INJECTION_RE.search(q):
        raise UnsafePrompt(
            "I can only answer questions about your uploaded documents. "
            "Please rephrase your question."
        )
    return q


def sanitize_context(text: str) -> str:
    """Neutralise document text before it enters the prompt (indirect injection)."""
    text = re.sub(_CONTROL_CHARS, "", text)
    text = re.sub(r"<\|?[a-z_]+\|?>", "", text, flags=re.I)
    return text.replace("</context>", "").replace("<context>", "")


# ------------------------------------------------------------------ headers
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    CSP = (
        "default-src 'self'; script-src 'self' 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline'; img-src 'self' data:; "
        "connect-src 'self'; font-src 'self'; object-src 'none'; "
        "base-uri 'none'; form-action 'self'; frame-ancestors 'none'"
    )

    async def dispatch(self, request, call_next):
        resp = await call_next(request)
        h = resp.headers
        h["Content-Security-Policy"] = self.CSP
        h["X-Content-Type-Options"] = "nosniff"
        h["X-Frame-Options"] = "DENY"
        h["Referrer-Policy"] = "no-referrer"
        h["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        h["Cross-Origin-Opener-Policy"] = "same-origin"
        if request.url.path.startswith("/api/"):
            h["Cache-Control"] = "no-store"
        return resp


# ------------------------------------------------------------------ private vaults
SESSION_COOKIE = "vault_session"
_TENANT_RE = re.compile(r"^[0-9a-f]{32}$")


def make_session_cookie(tenant: str, secret: str) -> str:
    sig = hmac.new(secret.encode(), tenant.encode(), hashlib.sha256).hexdigest()[:32]
    return f"{tenant}.{sig}"


def read_session_cookie(value, secret: str):
    """Return the tenant id if the cookie is authentic, else None."""
    if not value or "." not in value:
        return None
    tenant, _, sig = value.rpartition(".")
    if not _TENANT_RE.match(tenant):
        return None
    expected = hmac.new(secret.encode(), tenant.encode(), hashlib.sha256).hexdigest()[:32]
    return tenant if hmac.compare_digest(sig, expected) else None


class TenantMiddleware(BaseHTTPMiddleware):
    """Gives every visitor a private, unguessable vault id (signed cookie)."""

    def __init__(self, app, secret: str, secure: bool):
        super().__init__(app)
        self.secret = secret
        self.secure = secure

    async def dispatch(self, request, call_next):
        if not request.url.path.startswith("/api/"):
            return await call_next(request)
        tenant = read_session_cookie(request.cookies.get(SESSION_COOKIE), self.secret)
        fresh = tenant is None
        if fresh:
            tenant = uuid.uuid4().hex
        request.state.tenant = tenant
        resp = await call_next(request)
        if fresh:
            resp.set_cookie(
                SESSION_COOKIE,
                make_session_cookie(tenant, self.secret),
                max_age=60 * 60 * 24 * 30,
                httponly=True,
                samesite="strict",
                secure=self.secure,
                path="/",
            )
        return resp