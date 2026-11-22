import asyncio

import numpy as np
import pytest
from pydantic import BaseModel, ConfigDict

from csfsm_guardrails import Candidate, Guardrail, MemoryStore, Policy, ToolRegistry
from csfsm_guardrails.engine import Engine
from csfsm_guardrails.errors import GuardBlocked, PolicyError, SessionNotFound
from csfsm_guardrails.types import Evidence


class Args(BaseModel):
    model_config = ConfigDict(extra="forbid")
    record_id: str


@pytest.mark.parametrize("output", ["CSFSM_DEMO_SECRET", {"password": "abcdefghijk"}])
def test_tool_output_is_checked_before_release(output, store):
    tools = ToolRegistry().register("lookup_record", lambda record_id: output, Args)
    s = Guardrail(tools=tools, store=store).session()
    s.trusted_event("verified")
    s.grant_tool("lookup_record", {"record_id": "one"})
    result = s.execute_tool("lookup_record", {"record_id": "one"}, execution_id="one")
    assert result.executed and not result.released and result.result is None


def test_tool_failure_does_not_return_exception_content():
    def fail(record_id):
        raise RuntimeError("private traceback content")

    tools = ToolRegistry().register("lookup_record", fail, Args)
    s = Guardrail(tools=tools).session()
    s.trusted_event("verified")
    s.grant_tool("lookup_record", {"record_id": "one"})
    result = s.execute_tool("lookup_record", {"record_id": "one"}, execution_id="one")
    assert result.error == "tool_execution_failed"
    assert "private traceback" not in result.model_dump_json()


def test_async_tool():
    async def lookup(record_id):
        return {"record_id": record_id}

    tools = ToolRegistry().register("lookup_record", lookup, Args)
    s = Guardrail(tools=tools).session()
    s.trusted_event("verified")
    s.grant_tool("lookup_record", {"record_id": "one"})
    assert asyncio.run(
        s.aexecute_tool("lookup_record", {"record_id": "one"}, execution_id="one")
    ).released


def test_revocation_between_reservation_and_invocation_prevents_call(registered):
    class RevokeAfterReserve(MemoryStore):
        armed = False

        def transact(self, sid, owner, fn, **kwargs):
            response = super().transact(sid, owner, fn, **kwargs)
            if self.armed and "decision" in response:
                self.armed = False

                def revoke(s):
                    s.protocol_state = "unverified"
                    s.grants = []
                    s.revision += 1
                    return {}

                super().transact(sid, owner, revoke)
            return response

    store = RevokeAfterReserve()
    tools, calls = registered
    s = Guardrail(store=store, tools=tools).session()
    s.trusted_event("verified")
    s.grant_tool("lookup_record", {"record_id": "one"})
    store.armed = True
    result = s.execute_tool("lookup_record", {"record_id": "one"}, execution_id="one")
    assert not result.executed and not calls


def test_allowance_gate_and_protocol_successors_have_exact_zeros():
    data = Policy.default().model_dump()
    data["protocol_states"].append("unsafe")
    data["protocol_rules"].append(
        dict(source="unverified", action="allow_input", event="tool_error", target="unsafe")
    )
    engine = Engine(Policy.model_validate(data))
    state = engine.new_session()
    decision = engine.evaluate(
        state,
        Candidate(stage="input", text="A normal question"),
        evidence=Evidence(observation="routine"),
    )
    assert decision.action_probabilities["allow_input"] == 0
    assert "unsafe_protocol_successor" in decision.reasons


def test_sample_mode_never_selects_masked_action():
    data = Policy.default().model_dump()
    data["selection"] = "sample"
    policy = Policy.model_validate(data)
    for _ in range(40):
        decision = Guardrail(policy).session().check_output("CSFSM_DEMO_SECRET")
        assert not decision.allowed and decision.action_probabilities["release"] == 0


def test_policy_mismatch_and_copies():
    g = Guardrail()
    s = g.session()
    policy = g.policy
    policy.risk_threshold = 0.1
    assert g.policy.risk_threshold != policy.risk_threshold
    other = Guardrail(policy, store=g.store).resume(s.id)
    with pytest.raises(PolicyError):
        other.check_input("A normal request")


def test_custom_risk_blocks_actual_candidate_and_stop_remains_available():
    class Risk:
        def risks(self, candidate, state):
            p = Policy.default()
            values = np.array([a.risk for a in p.actions]).T
            if "prohibited" in candidate.text:
                values[:, [a.name for a in p.actions].index("release")] = 1
            return values

    s = Guardrail(risk_provider=Risk()).session()
    decision = s.check_output("A prohibited candidate")
    assert not decision.allowed and decision.action_probabilities["release"] == 0
    with pytest.raises(GuardBlocked):
        decision.require_allowed()


def test_store_transaction_rolls_back_on_failure(store):
    s = Guardrail(store=store).session()
    before = s.state()

    def fail(state):
        state.revision += 1
        raise RuntimeError("rollback")

    with pytest.raises(RuntimeError):
        store.transact(s.id, "local", fail)
    assert s.state() == before
    store.delete(s.id, "local")
    with pytest.raises(SessionNotFound):
        s.state()
