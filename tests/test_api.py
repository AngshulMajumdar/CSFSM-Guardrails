import asyncio

import httpx
import pytest
from fastapi.testclient import TestClient

from csfsm_guardrails import Candidate, Guardrail
from csfsm_guardrails.api import APISettings, Principal, create_app
from csfsm_guardrails.client import AsyncClient, Client

CLIENT = "client-test-key-longer-than-24-characters"
AUTHORITY = "authority-test-key-longer-than-24-characters"
OTHER = "other-test-key-longer-than-24-characters"


@pytest.fixture
def app(registered):
    tools, _ = registered
    settings = APISettings(
        keys={
            CLIENT: Principal(tenant="a"),
            AUTHORITY: Principal(tenant="a", role="authority"),
            OTHER: Principal(tenant="b"),
        },
        requests_per_minute=1000,
    )
    return create_app(Guardrail(tools=tools), settings)


def headers(key=CLIENT):
    return {"Authorization": f"Bearer {key}"}


def test_api_auth_roles_tenants_and_tool_path(app, registered):
    _, calls = registered
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert client.post("/v1/sessions").status_code == 401
        sid = client.post("/v1/sessions", headers=headers()).json()["id"]
        url = f"/v1/sessions/{sid}"
        assert client.get(url, headers=headers(OTHER)).status_code == 404
        assert (
            client.post(url + "/events", json={"event": "verified"}, headers=headers()).status_code
            == 403
        )
        assert (
            client.post(
                url + "/events", json={"event": "verified"}, headers=headers(AUTHORITY)
            ).status_code
            == 200
        )
        grant = {"tool": "lookup_record", "arguments": {"record_id": "one"}}
        assert (
            client.post(url + "/grants", json=grant, headers=headers(AUTHORITY)).status_code == 200
        )
        payload = dict(grant, execution_id="one")
        first = client.post(url + "/tools/execute", json=payload, headers=headers())
        assert first.status_code == 200 and first.json()["released"]
        assert (
            client.post(url + "/tools/execute", json=payload, headers=headers()).json()
            == first.json()
        )
        assert calls == ["one"]


def test_api_check_idempotency_and_forbidden_state_injection(app):
    with TestClient(app) as client:
        sid = client.post("/v1/sessions", headers=headers()).json()["id"]
        url = f"/v1/sessions/{sid}/check"
        h = dict(headers(), **{"Idempotency-Key": "same"})
        payload = {
            "candidate": {"stage": "input", "text": "Explain a state machine"},
            "expected_revision": 0,
        }
        one = client.post(url, json=payload, headers=h)
        assert one.status_code == 200
        assert client.post(url, json=payload, headers=h).json() == one.json()
        assert client.post(url, json=payload, headers=headers()).status_code == 409
        payload["candidate"]["verified"] = True
        assert client.post(url, json=payload, headers=headers()).status_code == 422


def test_body_limit_rate_limit_and_secure_startup():
    with pytest.raises(ValueError):
        create_app(Guardrail(), APISettings())
    app = create_app(
        Guardrail(), APISettings(keys={CLIENT: Principal(tenant="a")}, requests_per_minute=1)
    )
    with TestClient(app) as client:
        assert client.post("/v1/sessions", headers=headers()).status_code == 201
        assert client.post("/v1/sessions", headers=headers()).status_code == 429
        assert (
            client.post("/v1/sessions", content="x" * 70000, headers=headers()).status_code == 413
        )


def test_sync_client(app):
    with TestClient(app) as test:

        def handle(request):
            response = test.request(
                request.method, str(request.url), headers=request.headers, content=request.content
            )
            return httpx.Response(
                response.status_code, content=response.content, headers=dict(response.headers)
            )

        transport = httpx.MockTransport(handle)
        with Client("http://testserver", CLIENT, transport=transport) as client:
            sid = client.create_session()["id"]
            assert client.check(
                sid, Candidate(stage="input", text="Explain a state machine")
            ).allowed
            assert client.get_session(sid)["id"] == sid
            assert client.audit(sid)["events"]
            assert client.policy()["digest"]
            client.delete_session(sid)


def test_async_client(app):
    async def run():
        async with AsyncClient(
            "http://testserver", CLIENT, transport=httpx.ASGITransport(app=app)
        ) as client:
            sid = (await client.create_session())["id"]
            assert (
                await client.check(sid, Candidate(stage="input", text="Explain a state machine"))
            ).allowed
            assert (await client.get_session(sid))["id"] == sid
            assert (await client.audit(sid))["events"]
            assert (await client.policy())["digest"]
            await client.delete_session(sid)

    asyncio.run(run())
