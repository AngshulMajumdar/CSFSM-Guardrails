"""FastAPI factory. Credentials and authority roles are configured by the host."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import threading
import time
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import Field

from . import __version__
from .errors import (
    CapacityExceeded,
    ExecutionPending,
    GuardrailError,
    IdempotencyConflict,
    PolicyError,
    RevisionConflict,
    SessionNotFound,
)
from .guardrail import Guardrail
from .policy import Policy
from .store import SQLiteStore
from .types import Candidate, Decision, StrictModel, ToolResult


class Principal(StrictModel):
    tenant: str = Field(min_length=1, max_length=128)
    role: Literal["client", "authority"] = "client"


class APISettings(StrictModel):
    keys: dict[str, Principal] = Field(default_factory=dict, repr=False)
    allow_anonymous: bool = False
    requests_per_minute: int = Field(default=120, ge=1, le=100000)
    max_body_bytes: int = Field(default=65536, ge=1024, le=1048576)
    database: str = "./data/csfsm.sqlite3"

    @classmethod
    def from_env(cls):
        return cls(
            keys=json.loads(os.getenv("CSFSM_API_KEYS", "{}")),
            allow_anonymous=os.getenv("CSFSM_ALLOW_ANONYMOUS") == "1",
            database=os.getenv("CSFSM_DATABASE", "./data/csfsm.sqlite3"),
            requests_per_minute=int(os.getenv("CSFSM_RATE_LIMIT", "120")),
        )


class CheckRequest(StrictModel):
    candidate: Candidate
    expected_revision: int | None = Field(default=None, ge=0)


class EventRequest(StrictModel):
    event: str = Field(max_length=64)


class GrantRequest(StrictModel):
    tool: str = Field(min_length=1, max_length=128)
    arguments: dict = Field(default_factory=dict)
    ttl_seconds: int = Field(default=300, ge=1, le=3600)


class ExecuteRequest(StrictModel):
    tool: str = Field(min_length=1, max_length=128)
    arguments: dict = Field(default_factory=dict)
    execution_id: str = Field(min_length=1, max_length=128)
    expected_revision: int | None = Field(default=None, ge=0)


class BodyLimitMiddleware:
    def __init__(self, app, limit):
        self.app, self.limit = app, limit

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        # Buffer a bounded body before any endpoint or parser can act on it.
        messages, total = [], 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            total += len(message.get("body", b""))
            if total > self.limit:
                await JSONResponse({"error": "body_too_large"}, status_code=413)(
                    scope, receive, send
                )
                return
            messages.append(message)
            if not message.get("more_body", False):
                break

        async def replay():
            return messages.pop(0) if messages else await receive()

        await self.app(scope, replay, send)


def create_app(guardrail: Guardrail | None = None, settings: APISettings | None = None):
    settings = settings or APISettings.from_env()
    if not settings.keys and not settings.allow_anonymous:
        raise ValueError(
            "Set CSFSM_API_KEYS, or explicitly set CSFSM_ALLOW_ANONYMOUS=1 for local development"
        )
    if any(len(key) < 24 for key in settings.keys):
        raise ValueError("API keys must be at least 24 characters")
    if guardrail is None:
        policy_path = os.getenv("CSFSM_POLICY")
        guardrail = Guardrail(
            Policy.from_file(policy_path) if policy_path else None,
            store=SQLiteStore(settings.database),
        )
    app = FastAPI(
        title="CSFSM Guardrails API",
        version=__version__,
        description="Stateful conversational guardrails. Check endpoints return decisions; execute endpoints mediate registered tools.",
    )
    app.add_middleware(BodyLimitMiddleware, limit=settings.max_body_bytes)
    app.state.guardrail = guardrail
    buckets, lock = {}, threading.Lock()
    credential_hashes = [(hashlib.sha256(k.encode()).digest(), v) for k, v in settings.keys.items()]

    def authenticate(authorization: str | None = Header(default=None)):
        if authorization is None and settings.allow_anonymous:
            principal = Principal(tenant="local-development", role="client")
        else:
            if not authorization or not authorization.startswith("Bearer "):
                raise HTTPException(401, "Bearer API key required")
            digest = hashlib.sha256(authorization[7:].encode()).digest()
            principal = next(
                (p for expected, p in credential_hashes if hmac.compare_digest(digest, expected)),
                None,
            )
            if principal is None:
                raise HTTPException(401, "Invalid API key")
        now = time.monotonic()
        with lock:
            start, count = buckets.get(principal.tenant, (now, 0))
            if now - start >= 60:
                start, count = now, 0
            if count >= settings.requests_per_minute:
                raise HTTPException(429, "Request rate exceeded", headers={"Retry-After": "60"})
            buckets[principal.tenant] = (start, count + 1)
        return principal

    def authority(p=Depends(authenticate)):
        if p.role != "authority":
            raise HTTPException(403, "Trusted authority credential required")
        return p

    def idem(value: str | None):
        if value is not None and (not value or len(value) > 128):
            raise HTTPException(422, "Idempotency-Key must contain 1 to 128 characters")
        return value

    @app.exception_handler(GuardrailError)
    async def guard_error(request: Request, exc: GuardrailError):
        status = 400
        if isinstance(exc, SessionNotFound):
            status = 404
        elif isinstance(
            exc, (RevisionConflict, IdempotencyConflict, ExecutionPending, PolicyError)
        ):
            status = 409
        elif isinstance(exc, CapacityExceeded):
            status = 503
        return JSONResponse({"error": exc.code, "detail": str(exc)}, status_code=status)

    @app.exception_handler(ValueError)
    async def invalid_value(request: Request, exc: ValueError):
        return JSONResponse(
            {"error": "invalid_request", "detail": "Invalid operation or arguments"},
            status_code=422,
        )

    @app.get("/health")
    def health():
        return {"status": "ok", "version": __version__}

    @app.get("/v1/policy")
    def policy(p=Depends(authenticate)):
        return {
            "policy": guardrail.policy.model_dump(),
            "digest": guardrail.policy.digest,
            "tool_schemas": guardrail.tools.schemas(),
        }

    @app.post("/v1/sessions", status_code=201)
    def create_session(p=Depends(authenticate)):
        return guardrail.session(owner=p.tenant).view()

    @app.get("/v1/sessions/{session_id}")
    def get_session(session_id: str, p=Depends(authenticate)):
        return guardrail.resume(session_id, p.tenant).view()

    @app.delete("/v1/sessions/{session_id}", status_code=204)
    def delete_session(session_id: str, p=Depends(authenticate)):
        guardrail.store.delete(session_id, p.tenant)

    @app.get("/v1/sessions/{session_id}/audit")
    def audit(session_id: str, p=Depends(authenticate)):
        state = guardrail.store.get(session_id, p.tenant)
        return {"events": [e.model_dump() for e in state.audit], "head_hash": state.audit_hash}

    @app.post("/v1/sessions/{session_id}/check", response_model=Decision)
    def check(
        session_id: str,
        body: CheckRequest,
        p=Depends(authenticate),
        idempotency_key: str | None = Header(default=None),
    ):
        return guardrail.resume(session_id, p.tenant).check(
            body.candidate,
            expected_revision=body.expected_revision,
            idempotency_key=idem(idempotency_key),
        )

    @app.post("/v1/sessions/{session_id}/events")
    def event(
        session_id: str,
        body: EventRequest,
        p=Depends(authority),
        idempotency_key: str | None = Header(default=None),
    ):
        if body.event not in guardrail.policy.trusted_events:
            raise HTTPException(422, "Unknown trusted event")
        return guardrail.resume(session_id, p.tenant).trusted_event(
            body.event, idempotency_key=idem(idempotency_key)
        )

    @app.post("/v1/sessions/{session_id}/grants")
    def grant(
        session_id: str,
        body: GrantRequest,
        p=Depends(authority),
        idempotency_key: str | None = Header(default=None),
    ):
        return guardrail.resume(session_id, p.tenant).grant_tool(
            body.tool,
            body.arguments,
            ttl_seconds=body.ttl_seconds,
            idempotency_key=idem(idempotency_key),
        )

    @app.post("/v1/sessions/{session_id}/tools/execute", response_model=ToolResult)
    def execute(session_id: str, body: ExecuteRequest, p=Depends(authenticate)):
        return guardrail.resume(session_id, p.tenant).execute_tool(
            body.tool,
            body.arguments,
            execution_id=body.execution_id,
            expected_revision=body.expected_revision,
        )

    return app
