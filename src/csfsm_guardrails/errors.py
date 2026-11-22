"""Stable exceptions shared by the local SDK and HTTP transport."""


class GuardrailError(Exception):
    code = "guardrail_error"


class PolicyError(GuardrailError):
    code = "invalid_policy"


class ImpossibleObservation(GuardrailError):
    code = "impossible_observation"


class SessionNotFound(GuardrailError):
    code = "session_not_found"


class RevisionConflict(GuardrailError):
    code = "revision_conflict"


class IdempotencyConflict(GuardrailError):
    code = "idempotency_conflict"


class ExecutionPending(GuardrailError):
    code = "execution_pending"


class CapacityExceeded(GuardrailError):
    code = "capacity_exceeded"


class GuardBlocked(GuardrailError):
    code = "guard_blocked"

    def __init__(self, decision):
        self.decision = decision
        super().__init__(f"Guardrail selected {decision.action}: {', '.join(decision.reasons)}")
