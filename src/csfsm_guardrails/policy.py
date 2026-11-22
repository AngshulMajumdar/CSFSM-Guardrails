"""Validated, declarative configuration; loading a policy never imports code."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path
from typing import Literal

import numpy as np
import yaml
from pydantic import Field, model_validator

from .types import Stage, StrictModel, fingerprint


class ActionSpec(StrictModel):
    name: str
    stages: list[Stage]
    utility: list[float]
    risk: list[float]
    message: str = ""


class ProtocolRule(StrictModel):
    source: str = "*"
    action: str = "*"
    event: str
    target: str


class PatternRule(StrictModel):
    code: str
    pattern: str
    stages: list[Stage] = Field(default_factory=lambda: ["input", "output", "tool"])


class ToolSpec(StrictModel):
    requires_grant: bool = True
    requires_verification: bool = True


class JointMatrix(StrictModel):
    protocol_state: str
    previous_action: str
    event: str
    observation: str
    matrix: list[list[float]]


class Policy(StrictModel):
    name: str = "default"
    version: str = "1"
    states: list[str]
    observations: list[str]
    prior: list[float]
    transitions: dict[str, list[list[float]]]
    emissions: list[list[float]]
    joint_kernels: list[JointMatrix] = Field(default_factory=list)
    actions: list[ActionSpec]
    protocol_states: list[str] = Field(default_factory=lambda: ["unverified", "verified", "closed"])
    initial_protocol_state: str = "unverified"
    safe_protocol_states: list[str] = Field(
        default_factory=lambda: ["unverified", "verified", "closed"]
    )
    closed_state: str = "closed"
    verified_states: list[str] = Field(default_factory=lambda: ["verified"])
    trusted_events: list[str] = Field(
        default_factory=lambda: ["none", "verified", "revoked", "tool_success", "tool_error"]
    )
    protocol_rules: list[ProtocolRule] = Field(default_factory=list)
    patterns: list[PatternRule] = Field(default_factory=list)
    tools: dict[str, ToolSpec] = Field(default_factory=dict)
    temperature: float = Field(default=0.7, gt=0, le=100)
    risk_threshold: float = Field(default=0.20, ge=0, le=1)
    uncertainty_radius: float = Field(default=0.02, ge=0, le=1)
    selection: Literal["max", "sample"] = "max"
    risk_semantics: Literal["model_score", "conditional_bound"] = "model_score"
    envelope_description: str | None = None
    session_alpha: float | None = Field(default=None, ge=0, le=1)
    max_decisions: int = Field(default=100, ge=1, le=10000)
    max_clarifications: int = Field(default=3, ge=0, le=100)
    session_ttl_seconds: int = Field(default=3600, ge=1, le=86400 * 7)

    @model_validator(mode="after")
    def check_model(self):
        def unique(v, name):
            if not v or len(set(v)) != len(v):
                raise ValueError(f"{name} must be nonempty and unique")

        def array(v, shape, name, rows=False):
            a = np.asarray(v, dtype=float)
            if a.shape != shape or not np.isfinite(a).all() or (a < 0).any():
                raise ValueError(f"Invalid shape or entries in {name}; expected {shape}")
            if rows and not np.allclose(a.sum(axis=-1), 1, atol=1e-10, rtol=0):
                raise ValueError(f"{name} must be row stochastic")
            return a

        n, m = len(self.states), len(self.observations)
        if n > 256 or m > 256 or len(self.actions) > 64:
            raise ValueError("Policy exceeds supported dimensions")
        unique(self.states, "states")
        unique(self.observations, "observations")
        unique(self.protocol_states, "protocol_states")
        unique(self.trusted_events, "trusted_events")
        names = [a.name for a in self.actions]
        unique(names, "actions")
        for required in ["allow_input", "release", "execute_tool", "stop"]:
            if required not in names:
                raise ValueError(f"Missing required action {required}")
        array(self.prior, (n,), "prior", rows=True)
        array(self.emissions, (n, m), "emissions", rows=True)
        if "default" not in self.transitions:
            raise ValueError("A default transition matrix is required")
        for key, v in self.transitions.items():
            if key not in names + ["default", "noop"]:
                raise ValueError("Unknown transition action")
            array(v, (n, n), f"transitions.{key}", rows=True)
        for a in self.actions:
            array(a.risk, (n,), f"risk.{a.name}")
            if max(a.risk) > 1 or len(a.utility) != n or not np.isfinite(a.utility).all():
                raise ValueError("Invalid risk or utility")
        stop = next(a for a in self.actions if a.name == "stop")
        if any(stop.risk) or set(stop.stages) != {"input", "output", "tool"}:
            raise ValueError("Stop must have zero declared risk and exist in every stage")
        for v in (
            self.safe_protocol_states
            + self.verified_states
            + [self.initial_protocol_state, self.closed_state]
        ):
            if v not in self.protocol_states:
                raise ValueError("Unknown protocol state")
        if (
            self.closed_state not in self.safe_protocol_states
            or self.initial_protocol_state not in self.safe_protocol_states
        ):
            raise ValueError("Initial and closed states must be safe")
        keys = set()
        for r in self.protocol_rules:
            key = (r.source, r.action, r.event)
            if (
                key in keys
                or r.target not in self.protocol_states
                or r.event not in self.trusted_events
            ):
                raise ValueError("Duplicate or invalid protocol rule")
            if (
                r.source not in ["*"] + self.protocol_states
                or r.action not in ["*", "noop"] + names
            ):
                raise ValueError("Invalid protocol rule source/action")
            keys.add(key)
        # No two equally specific wildcard rules may disagree at runtime.
        for s in self.protocol_states:
            for a in names + ["noop"]:
                for e in self.trusted_events:
                    self.successor(s, a, e)
        groups = {}
        joint_keys = set()
        for k in self.joint_kernels:
            key = (k.protocol_state, k.previous_action, k.event, k.observation)
            if (
                key in joint_keys
                or k.protocol_state not in self.protocol_states
                or k.previous_action not in names + ["noop"]
                or k.event not in self.trusted_events
                or k.observation not in self.observations
            ):
                raise ValueError("Invalid or duplicate joint kernel key")
            joint_keys.add(key)
            v = array(k.matrix, (n, n), "joint kernel")
            group = (k.protocol_state, k.previous_action)
            groups[group] = groups.get(group, np.zeros(n)) + v.sum(axis=1)
        if self.joint_kernels:
            expected = {(s, a) for s in self.protocol_states for a in names + ["noop"]}
            if set(groups) != expected or any(
                not np.allclose(v, 1, atol=1e-10, rtol=0) for v in groups.values()
            ):
                raise ValueError(
                    "Joint kernels need every protocol/action group normalized across events and observations"
                )
        if self.risk_semantics == "conditional_bound" and not self.envelope_description:
            raise ValueError("Conditional bounds require a stated, externally justified envelope")
        if self.session_alpha is not None and self.risk_semantics != "conditional_bound":
            raise ValueError("A session probability budget requires conditional_bound mode")
        if (
            "neutral" not in self.observations
            or "unknown" not in self.observations
            or "none" not in self.trusted_events
        ):
            raise ValueError("neutral/unknown observations and none event are required")
        import re

        for rule in self.patterns:
            re.compile(rule.pattern)
        return self

    def successor(self, state: str, action: str, event: str) -> str:
        if state == self.closed_state or action == "stop":
            return self.closed_state
        if state not in self.protocol_states or event not in self.trusted_events:
            return self.closed_state
        matches = [
            (int(r.source != "*") + int(r.action != "*"), r.target)
            for r in self.protocol_rules
            if r.source in (state, "*") and r.action in (action, "*") and r.event == event
        ]
        if not matches:
            return state
        top = max(x[0] for x in matches)
        targets = {target for score, target in matches if score == top}
        if len(targets) != 1:
            raise ValueError("Ambiguous protocol transitions")
        return targets.pop()

    @property
    def digest(self) -> str:
        return fingerprint(self.model_dump())

    @classmethod
    def from_file(cls, path: str | Path) -> "Policy":
        raw = Path(path).read_text()
        if len(raw.encode()) > 4 * 1024 * 1024:
            raise ValueError("Policy file exceeds 4 MiB")
        return cls.model_validate(yaml.safe_load(raw))

    @classmethod
    def default(cls) -> "Policy":
        return cls.model_validate(
            yaml.safe_load(files("csfsm_guardrails.policies").joinpath("default.yaml").read_text())
        )
