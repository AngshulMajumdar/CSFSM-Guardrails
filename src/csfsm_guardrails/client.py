"""Synchronous and asynchronous HTTP clients. Transport errors never mean allow."""

import uuid

import httpx

from .types import Candidate, Decision, ToolResult


class Client:
    def __init__(self, base_url, api_key, *, timeout=30.0, transport=None):
        self.http = httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=timeout,
            transport=transport,
            headers={"Authorization": f"Bearer {api_key}"},
        )

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        self.http.close()

    def create_session(self):
        r = self.http.post("/v1/sessions")
        r.raise_for_status()
        return r.json()

    def get_session(self, session_id):
        r = self.http.get(f"/v1/sessions/{session_id}")
        r.raise_for_status()
        return r.json()

    def delete_session(self, session_id):
        self.http.delete(f"/v1/sessions/{session_id}").raise_for_status()

    def audit(self, session_id):
        r = self.http.get(f"/v1/sessions/{session_id}/audit")
        r.raise_for_status()
        return r.json()

    def policy(self):
        r = self.http.get("/v1/policy")
        r.raise_for_status()
        return r.json()

    def trusted_event(self, session_id, event, *, idempotency_key=None):
        r = self.http.post(
            f"/v1/sessions/{session_id}/events",
            json={"event": event},
            headers={"Idempotency-Key": idempotency_key or str(uuid.uuid4())},
        )
        r.raise_for_status()
        return r.json()

    def grant_tool(self, session_id, tool, arguments, *, ttl_seconds=300, idempotency_key=None):
        r = self.http.post(
            f"/v1/sessions/{session_id}/grants",
            json=dict(tool=tool, arguments=arguments, ttl_seconds=ttl_seconds),
            headers={"Idempotency-Key": idempotency_key or str(uuid.uuid4())},
        )
        r.raise_for_status()
        return r.json()

    def check(
        self, session_id, candidate: Candidate, *, expected_revision=None, idempotency_key=None
    ):
        r = self.http.post(
            f"/v1/sessions/{session_id}/check",
            json={"candidate": candidate.model_dump(), "expected_revision": expected_revision},
            headers={"Idempotency-Key": idempotency_key or str(uuid.uuid4())},
        )
        r.raise_for_status()
        return Decision.model_validate(r.json())

    def execute_tool(self, session_id, tool, arguments, *, execution_id, expected_revision=None):
        r = self.http.post(
            f"/v1/sessions/{session_id}/tools/execute",
            json=dict(
                tool=tool,
                arguments=arguments,
                execution_id=execution_id,
                expected_revision=expected_revision,
            ),
        )
        r.raise_for_status()
        return ToolResult.model_validate(r.json())


class AsyncClient:
    def __init__(self, base_url, api_key, *, timeout=30.0, transport=None):
        self.http = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=timeout,
            transport=transport,
            headers={"Authorization": f"Bearer {api_key}"},
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()

    async def close(self):
        await self.http.aclose()

    async def create_session(self):
        r = await self.http.post("/v1/sessions")
        r.raise_for_status()
        return r.json()

    async def get_session(self, session_id):
        r = await self.http.get(f"/v1/sessions/{session_id}")
        r.raise_for_status()
        return r.json()

    async def delete_session(self, session_id):
        r = await self.http.delete(f"/v1/sessions/{session_id}")
        r.raise_for_status()

    async def audit(self, session_id):
        r = await self.http.get(f"/v1/sessions/{session_id}/audit")
        r.raise_for_status()
        return r.json()

    async def policy(self):
        r = await self.http.get("/v1/policy")
        r.raise_for_status()
        return r.json()

    async def trusted_event(self, session_id, event, *, idempotency_key=None):
        r = await self.http.post(
            f"/v1/sessions/{session_id}/events",
            json={"event": event},
            headers={"Idempotency-Key": idempotency_key or str(uuid.uuid4())},
        )
        r.raise_for_status()
        return r.json()

    async def grant_tool(
        self, session_id, tool, arguments, *, ttl_seconds=300, idempotency_key=None
    ):
        r = await self.http.post(
            f"/v1/sessions/{session_id}/grants",
            json=dict(tool=tool, arguments=arguments, ttl_seconds=ttl_seconds),
            headers={"Idempotency-Key": idempotency_key or str(uuid.uuid4())},
        )
        r.raise_for_status()
        return r.json()

    async def check(
        self, session_id, candidate: Candidate, *, expected_revision=None, idempotency_key=None
    ):
        r = await self.http.post(
            f"/v1/sessions/{session_id}/check",
            json={"candidate": candidate.model_dump(), "expected_revision": expected_revision},
            headers={"Idempotency-Key": idempotency_key or str(uuid.uuid4())},
        )
        r.raise_for_status()
        return Decision.model_validate(r.json())

    async def execute_tool(
        self, session_id, tool, arguments, *, execution_id, expected_revision=None
    ):
        r = await self.http.post(
            f"/v1/sessions/{session_id}/tools/execute",
            json=dict(
                tool=tool,
                arguments=arguments,
                execution_id=execution_id,
                expected_revision=expected_revision,
            ),
        )
        r.raise_for_status()
        return ToolResult.model_validate(r.json())
