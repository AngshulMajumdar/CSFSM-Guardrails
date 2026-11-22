"""Public Python SDK, including enforcing generation and tool wrappers."""

from __future__ import annotations

import asyncio
from typing import Any

from .engine import Engine
from .errors import ExecutionPending, IdempotencyConflict
from .policy import Policy
from .store import MemoryStore
from .tools import ToolRegistry
from .types import Candidate, Decision, GuardedResult, ToolResult, canonical_json, fingerprint


class Guardrail:
    def __init__(
        self,
        policy: Policy | None = None,
        *,
        store=None,
        encoder=None,
        validators=(),
        risk_provider=None,
        tools=None,
        clock=None,
    ):
        options = {} if clock is None else {"clock": clock}
        self.engine = Engine(
            policy, encoder=encoder, validators=validators, risk_provider=risk_provider, **options
        )
        self.store = store or MemoryStore(**options)
        self.tools = tools or ToolRegistry()

    @property
    def policy(self):
        return self.engine.policy.model_copy(deep=True)

    def session(self, owner="local") -> "Session":
        state = self.engine.new_session(owner)
        self.store.create(state)
        return Session(self, state.id, owner)

    def resume(self, session_id, owner="local") -> "Session":
        self.store.get(session_id, owner)
        return Session(self, session_id, owner)


class Session:
    def __init__(self, guard: Guardrail, session_id: str, owner: str):
        self.guard, self.id, self.owner = guard, session_id, owner

    def state(self):
        return self.guard.store.get(self.id, self.owner)

    def view(self):
        s = self.state()
        return dict(
            id=s.id,
            revision=s.revision,
            protocol_state=s.protocol_state,
            belief=dict(zip(self.guard.engine.policy.states, s.belief)),
            decisions=s.decisions,
            observations=s.observations,
            clarifications=s.clarifications,
            expires_at=s.expires_at,
            policy_digest=s.policy_digest,
            consumed_allowance=s.consumed_allowance,
            audit_hash=s.audit_hash,
        )

    def check(
        self, candidate: Candidate, *, expected_revision=None, idempotency_key=None
    ) -> Decision:
        response = self.guard.store.transact(
            self.id,
            self.owner,
            lambda s: self.guard.engine.evaluate(s, candidate).model_dump(),
            expected_revision=expected_revision,
            idempotency_key=idempotency_key,
            request_digest=fingerprint({"operation": "check", "candidate": candidate.model_dump()}),
        )
        return Decision.model_validate(response)

    def check_input(self, text, **kwargs):
        return self.check(Candidate(stage="input", text=text), **kwargs)

    def check_output(self, text, **kwargs):
        return self.check(Candidate(stage="output", text=text), **kwargs)

    def check_tool(self, name, arguments, **kwargs):
        return self.check(Candidate(stage="tool", tool=name, arguments=arguments), **kwargs)

    def trusted_event(self, event, *, idempotency_key=None):
        def change(s):
            self.guard.engine.trusted_event(s, event)
            return {"revision": s.revision, "protocol_state": s.protocol_state}

        return self.guard.store.transact(
            self.id,
            self.owner,
            change,
            idempotency_key=idempotency_key,
            request_digest=fingerprint({"event": event}),
        )

    def grant_tool(self, name, arguments, *, ttl_seconds=300, idempotency_key=None):
        # Normalize known schemas before binding an authorization to arguments.
        if name in self.guard.tools.schemas():
            arguments = self.guard.tools.get(name).normalize(arguments)
        Candidate(stage="tool", tool=name, arguments=arguments)
        return self.guard.store.transact(
            self.id,
            self.owner,
            lambda s: self.guard.engine.grant_tool(s, name, arguments, ttl_seconds).model_dump(),
            idempotency_key=idempotency_key,
            request_digest=fingerprint({"grant": name, "arguments": arguments, "ttl": ttl_seconds}),
        )

    def run(self, text: str, generate) -> GuardedResult:
        before = self.check_input(text)
        if not before.allowed:
            return GuardedResult(content=before.message, allowed=False, input_decision=before)
        draft = generate(text)
        if not isinstance(draft, str):
            raise TypeError("Generator must return text; tool calls require execute_tool")
        after = self.check_output(draft, expected_revision=before.revision)
        return GuardedResult(
            content=draft if after.allowed else after.message,
            allowed=after.allowed,
            input_decision=before,
            output_decision=after,
        )

    async def arun(self, text: str, generate) -> GuardedResult:
        before = await asyncio.to_thread(self.check_input, text)
        if not before.allowed:
            return GuardedResult(content=before.message, allowed=False, input_decision=before)
        draft = await generate(text)
        if not isinstance(draft, str):
            raise TypeError("Generator must return text")
        after = await asyncio.to_thread(self.check_output, draft, expected_revision=before.revision)
        return GuardedResult(
            content=draft if after.allowed else after.message,
            allowed=after.allowed,
            input_decision=before,
            output_decision=after,
        )

    def stream(self, text: str, generate_chunks):
        """Buffer and validate a complete response before releasing any text."""

        def collect(prompt):
            parts, size = [], 0
            for part in generate_chunks(prompt):
                if not isinstance(part, str):
                    raise TypeError("Stream chunks must be text")
                size += len(part)
                if size > 32000:
                    raise ValueError("Buffered response exceeds 32000 characters")
                parts.append(part)
            return "".join(parts)

        yield self.run(text, collect).content

    async def astream(self, text: str, generate_chunks):
        async def collect(prompt):
            parts, size = [], 0
            async for part in generate_chunks(prompt):
                if not isinstance(part, str):
                    raise TypeError("Stream chunks must be text")
                size += len(part)
                if size > 32000:
                    raise ValueError("Buffered response exceeds 32000 characters")
                parts.append(part)
            return "".join(parts)

        yield (await self.arun(text, collect)).content

    def execute_tool(
        self, name: str, arguments: dict[str, Any], *, execution_id: str, expected_revision=None
    ) -> ToolResult:
        """Reserve durably before invoking a registered function.

        An interrupted reservation is never auto-replayed. The host must reconcile
        unknown side effects. This is not a distributed exactly-once protocol.
        """
        if not execution_id or len(execution_id) > 128:
            raise ValueError("A nonempty execution_id of at most 128 characters is required")
        tool = self.guard.tools.get(name)
        arguments = tool.normalize(arguments)
        candidate = Candidate(stage="tool", tool=name, arguments=arguments)
        digest = candidate.digest
        engine = self.guard.engine

        def reserve(s):
            previous = s.executions.get(execution_id)
            if previous:
                if previous["digest"] != digest:
                    raise IdempotencyConflict("Execution ID belongs to different arguments")
                if previous["status"] == "running":
                    raise ExecutionPending(
                        "Execution may already have occurred; reconcile before retrying"
                    )
                # Releasing a cached result must not bypass revocation or expiry.
                grant_id = previous.get("grant_id")
                expired = previous.get("authorization_expires_at", 0) <= engine.clock()
                revoked = grant_id is not None and not any(g.id == grant_id for g in s.grants)
                spec = engine.policy.tools[name]
                if (
                    s.protocol_state == engine.policy.closed_state
                    or expired
                    or revoked
                    or (
                        spec.requires_verification
                        and s.protocol_state not in engine.policy.verified_states
                    )
                ):
                    raise ValueError("Cached result authorization is no longer valid")
                return {"cached": previous["result"]}
            decision = engine.evaluate(s, candidate)
            if not decision.allowed:
                return {
                    "blocked": ToolResult(
                        execution_id=execution_id, executed=False, released=False, decision=decision
                    ).model_dump()
                }
            grant = engine._matching_grant(s, candidate)
            engine.consume_permission(s, candidate)
            s.executions[execution_id] = {
                "digest": digest,
                "status": "running",
                "decision": decision.model_dump(),
                "grant_id": grant.id if grant else None,
                "authorization_expires_at": grant.expires_at if grant else s.expires_at,
            }
            return {"decision": decision.model_dump()}

        reservation = self.guard.store.transact(
            self.id, self.owner, reserve, expected_revision=expected_revision
        )
        if "cached" in reservation:
            return ToolResult.model_validate(reservation["cached"])
        if "blocked" in reservation:
            return ToolResult.model_validate(reservation["blocked"])
        decision = Decision.model_validate(reservation["decision"])

        def complete(s):
            # The pending reservation is already durable. Serialize permission
            # changes with invocation and result release on this store.
            record = s.executions[execution_id]
            spec = engine.policy.tools[name]

            def authorized():
                return (
                    record["authorization_expires_at"] > engine.clock()
                    and s.expires_at > engine.clock()
                    and s.protocol_state != engine.policy.closed_state
                    and (
                        not spec.requires_verification
                        or s.protocol_state in engine.policy.verified_states
                    )
                    and (
                        record["grant_id"] is None
                        or any(g.id == record["grant_id"] for g in s.grants)
                    )
                )

            if not authorized():
                result = ToolResult(
                    execution_id=execution_id,
                    executed=False,
                    released=False,
                    decision=decision,
                    error="authorization_expired_or_revoked",
                )
                record.update(status="done", result=result.model_dump())
                return result.model_dump()
            value, error = None, None
            try:
                value = tool.invoke(arguments)
                serialized = value if isinstance(value, str) else canonical_json(value)
                if len(serialized) > 32000:
                    raise ValueError("Tool output too large")
            except Exception:
                error = "tool_execution_failed"
            engine.trusted_event(s, "tool_error" if error else "tool_success")
            output_decision = None
            released = False
            if not error:
                # Check permission freshness again before disclosing the result.
                if authorized():
                    output_decision = engine.evaluate(s, Candidate(stage="output", text=serialized))
                    released = output_decision.allowed
            result = ToolResult(
                execution_id=execution_id,
                executed=True,
                released=released,
                result=value if released else None,
                decision=decision,
                output_decision=output_decision,
                error=error or (None if released else "tool_result_withheld"),
            )
            s.executions[execution_id].update(status="done", result=result.model_dump())
            return result.model_dump()

        return ToolResult.model_validate(self.guard.store.transact(self.id, self.owner, complete))

    async def aexecute_tool(self, name, arguments, **kwargs):
        return await asyncio.to_thread(self.execute_tool, name, arguments, **kwargs)
