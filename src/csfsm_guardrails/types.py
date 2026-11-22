"""Typed public values. User content never supplies trusted state or grants."""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Stage = Literal["input", "output", "tool"]
PERMIT_ACTION = {"input": "allow_input", "output": "release", "tool": "execute_tool"}


def canonical_json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, allow_inf_nan=False)


class Candidate(StrictModel):
    stage: Stage
    text: str = Field(default="", max_length=32000)
    tool: str | None = Field(default=None, max_length=128)
    arguments: dict[str, Any] = Field(default_factory=dict)

    @field_validator("arguments")
    @classmethod
    def bounded_arguments(cls, value):
        encoded = canonical_json(value)
        if len(encoded.encode()) > 32768:
            raise ValueError("Tool arguments exceed 32 KiB")
        return value

    @property
    def digest(self) -> str:
        return fingerprint(self.model_dump())


class Evidence(StrictModel):
    observation: str
    reasons: list[str] = Field(default_factory=list)
    source: str = "custom"


class Finding(StrictModel):
    code: str
    blocks: bool = True


class Grant(StrictModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tool: str
    arguments_digest: str
    expires_at: float
    used: bool = False


class AuditEvent(StrictModel):
    sequence: int
    timestamp: float
    kind: str
    action: str
    payload_digest: str
    previous_hash: str
    event_hash: str
    reasons: list[str] = Field(default_factory=list)


class SessionState(StrictModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    owner: str
    policy_digest: str
    protocol_state: str
    belief: list[float]
    created_at: float = Field(default_factory=time.time)
    expires_at: float
    revision: int = 0
    observations: int = 0
    decisions: int = 0
    clarifications: int = 0
    last_action: str = "noop"
    consumed_allowance: float = 0.0
    grants: list[Grant] = Field(default_factory=list)
    audit: list[AuditEvent] = Field(default_factory=list)
    audit_sequence: int = 0
    audit_hash: str = "0" * 64
    executions: dict[str, dict[str, Any]] = Field(default_factory=dict)


class Decision(StrictModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    revision: int
    stage: Stage
    action: str
    allowed: bool
    message: str
    reasons: list[str]
    action_probabilities: dict[str, float]
    nominal_risks: dict[str, float]
    upper_risks: dict[str, float]
    blocked_actions: dict[str, list[str]]
    belief: dict[str, float]
    protocol_state: str
    candidate_digest: str
    policy_digest: str
    risk_semantics: Literal["model_score", "conditional_bound"]
    risk_allowance: float
    consumed_allowance: float
    observation: str | None = None

    def require_allowed(self) -> "Decision":
        if not self.allowed:
            from .errors import GuardBlocked

            raise GuardBlocked(self)
        return self


class GuardedResult(StrictModel):
    content: str
    allowed: bool
    input_decision: Decision
    output_decision: Decision | None = None


class ToolResult(StrictModel):
    execution_id: str
    executed: bool
    released: bool
    result: Any = None
    decision: Decision
    output_decision: Decision | None = None
    error: str | None = None
