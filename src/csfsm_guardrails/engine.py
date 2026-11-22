"""Stateful enforcement logic; language evidence cannot mutate trusted permissions."""

from __future__ import annotations

import secrets
import time
from typing import Callable

import numpy as np

from .errors import ImpossibleObservation, PolicyError
from .extensions import PatternValidator, RuleBasedEncoder
from .math import forward, forward_joint, masked_softmax, robust_risk
from .policy import Policy
from .types import (
    PERMIT_ACTION,
    AuditEvent,
    Candidate,
    Decision,
    Evidence,
    Grant,
    SessionState,
    fingerprint,
)


class Engine:
    def __init__(
        self,
        policy: Policy | None = None,
        *,
        encoder=None,
        validators=(),
        risk_provider=None,
        clock: Callable[[], float] = time.time,
    ):
        self.policy = (policy or Policy.default()).model_copy(deep=True)
        self.encoder = encoder or RuleBasedEncoder()
        self.validators = [PatternValidator(self.policy.patterns), *validators]
        self.risk_provider = risk_provider
        self.clock = clock
        self.names = [a.name for a in self.policy.actions]
        self.joint = {
            (k.protocol_state, k.previous_action, k.event, k.observation): k.matrix
            for k in self.policy.joint_kernels
        }

    def new_session(self, owner="local") -> SessionState:
        return SessionState(
            owner=owner,
            policy_digest=self.policy.digest,
            protocol_state=self.policy.initial_protocol_state,
            belief=self.policy.prior.copy(),
            created_at=self.clock(),
            expires_at=self.clock() + self.policy.session_ttl_seconds,
        )

    def _check_policy(self, state):
        if state.policy_digest != self.policy.digest:
            raise PolicyError("Session belongs to a different policy revision")

    def _audit(self, state, kind, action, digest, reasons):
        state.audit_sequence += 1
        data = dict(
            sequence=state.audit_sequence,
            timestamp=self.clock(),
            kind=kind,
            action=action,
            payload_digest=digest,
            previous_hash=state.audit_hash,
            reasons=list(reasons),
        )
        hashed = fingerprint(data)
        state.audit.append(AuditEvent(**data, event_hash=hashed))
        state.audit = state.audit[-100:]
        state.audit_hash = hashed

    def observe(self, state, evidence: Evidence, *, trusted_event="none"):
        self._check_policy(state)
        if trusted_event not in self.policy.trusted_events:
            raise ImpossibleObservation("Unknown trusted event")
        if evidence.observation not in self.policy.observations:
            raise ImpossibleObservation("Unknown evidence symbol")
        old_s, old_a = state.protocol_state, state.last_action
        if self.joint:
            matrix = self.joint.get((old_s, old_a, trusted_event, evidence.observation))
            if matrix is None:
                raise ImpossibleObservation("Joint observation has zero probability")
            b = forward_joint(state.belief, matrix)
        else:
            transition = self.policy.transitions.get(old_a, self.policy.transitions["default"])
            col = self.policy.observations.index(evidence.observation)
            likelihood = np.asarray(self.policy.emissions)[:, col]
            b = forward(state.belief, transition, likelihood)
        state.belief = b.tolist()
        state.protocol_state = self.policy.successor(old_s, old_a, trusted_event)
        state.observations += 1

    def trusted_event(self, state, event: str):
        """Trusted host interface. Never expose this directly to user prompts."""
        self._check_policy(state)
        try:
            self.observe(
                state, Evidence(observation="neutral", source="trusted_host"), trusted_event=event
            )
        except ImpossibleObservation:
            state.protocol_state = self.policy.closed_state
            state.grants = []
        if event == "revoked":
            state.grants = []
        state.last_action = "noop"
        state.revision += 1
        self._audit(state, "trusted_event", event, fingerprint(event), [])

    def grant_tool(self, state, tool, arguments, ttl_seconds=300):
        self._check_policy(state)
        if state.protocol_state == self.policy.closed_state or tool not in self.policy.tools:
            raise ValueError("Cannot grant an unknown tool or closed session")
        if not 0 < ttl_seconds <= 3600:
            raise ValueError("Grant lifetime must be between 1 and 3600 seconds")
        # Drop consumed/expired grants, and bound outstanding authorization state.
        state.grants = [g for g in state.grants if not g.used and g.expires_at > self.clock()]
        if len(state.grants) >= 64:
            raise ValueError("Too many outstanding grants")
        grant = Grant(
            tool=tool,
            arguments_digest=fingerprint(arguments),
            expires_at=min(self.clock() + ttl_seconds, state.expires_at),
        )
        state.grants.append(grant)
        state.revision += 1
        self._audit(state, "grant", tool, fingerprint(arguments), ["request_bound", "single_use"])
        return grant

    def _matching_grant(self, state, candidate):
        digest = fingerprint(candidate.arguments)
        return next(
            (
                g
                for g in state.grants
                if g.tool == candidate.tool
                and g.arguments_digest == digest
                and not g.used
                and g.expires_at > self.clock()
            ),
            None,
        )

    def tool_reasons(self, state, candidate):
        spec = self.policy.tools.get(candidate.tool)
        if spec is None:
            return ["unknown_tool"]
        reasons = []
        if spec.requires_verification and state.protocol_state not in self.policy.verified_states:
            reasons.append("verification_required")
        if spec.requires_grant and self._matching_grant(state, candidate) is None:
            reasons.append("request_bound_grant_required")
        return reasons

    def consume_permission(self, state, candidate):
        # Called again on the execution path, immediately before the operation.
        if state.protocol_state == self.policy.closed_state or state.expires_at <= self.clock():
            raise ValueError("Session is closed or expired")
        reasons = self.tool_reasons(state, candidate)
        if reasons:
            raise ValueError("; ".join(reasons))
        if self.policy.tools[candidate.tool].requires_grant:
            self._matching_grant(state, candidate).used = True

    def evaluate(
        self, state: SessionState, candidate: Candidate, *, evidence: Evidence | None = None
    ) -> Decision:
        self._check_policy(state)
        permit = PERMIT_ACTION[candidate.stage]
        common, observation = [], None
        terminal = (
            state.protocol_state == self.policy.closed_state or state.expires_at <= self.clock()
        )
        if state.decisions >= self.policy.max_decisions:
            terminal = True
            common.append("decision_limit")
        if terminal:
            common.append("session_closed_or_expired")
        if candidate.stage == "input" and not terminal:
            try:
                evidence = evidence or self.encoder.encode(
                    candidate.text, state.model_copy(deep=True)
                )
                self.observe(state, evidence)
                observation = evidence.observation
            except Exception:
                terminal = True
                common.append("evidence_failure")
        findings = []
        if not terminal:
            try:
                for validator in self.validators:
                    findings.extend(validator.validate(candidate, state.model_copy(deep=True)))
                common.extend(f.code for f in findings if f.blocks)
            except Exception:
                terminal = True
                common.append("validator_failure")
        if candidate.stage == "output" and not candidate.text.strip():
            common.append("empty_output")
        if candidate.stage == "tool":
            common.extend(self.tool_reasons(state, candidate))
        allowance = self.policy.risk_threshold
        if self.policy.session_alpha is not None:
            allowance = min(allowance, self.policy.session_alpha / self.policy.max_decisions)
        try:
            matrix = (
                self.risk_provider.risks(candidate, state.model_copy(deep=True))
                if self.risk_provider
                else np.array([a.risk for a in self.policy.actions]).T
            )
            nominal, upper = robust_risk(state.belief, matrix, self.policy.uncertainty_radius)
            if len(nominal) != len(self.names):
                raise ValueError("Wrong action dimension")
            if upper[self.names.index("stop")] != 0:
                raise ValueError("Stop risk must remain zero")
        except Exception:
            terminal = True
            common.append("risk_provider_failure")
            nominal = np.zeros(len(self.names))
            upper = np.ones(len(self.names))
            upper[self.names.index("stop")] = 0
        utilities = np.array([np.dot(state.belief, a.utility) for a in self.policy.actions])
        blocked, mask = {}, []
        for idx, action in enumerate(self.policy.actions):
            reasons = []
            if action.name != "stop":
                if terminal:
                    reasons.append("terminal")
                if candidate.stage not in action.stages:
                    reasons.append("wrong_stage")
                if any(
                    self.policy.successor(state.protocol_state, action.name, event)
                    not in self.policy.safe_protocol_states
                    for event in self.policy.trusted_events
                ):
                    reasons.append("unsafe_protocol_successor")
                if action.name == permit:
                    reasons.extend(common)
                if (
                    action.name == "clarify"
                    and state.clarifications >= self.policy.max_clarifications
                ):
                    reasons.append("clarification_limit")
                if upper[idx] > allowance:
                    reasons.append("risk_threshold")
            mask.append(not reasons)
            if reasons:
                blocked[action.name] = sorted(set(reasons))
        if "request_verification" in self.names and any("required" in r for r in common):
            utilities[self.names.index("request_verification")] += 15
        if (
            common
            and not terminal
            and not any("required" in r for r in common)
            and "refuse" in self.names
        ):
            utilities[self.names.index("refuse")] += 15
        probs = masked_softmax(utilities, mask, self.policy.temperature)
        if self.policy.selection == "sample":
            idx = secrets.SystemRandom().choices(range(len(probs)), weights=probs, k=1)[0]
        else:
            idx = int(np.argmax(probs))
        selected = self.names[idx]
        state.last_action = selected
        if selected == "stop":
            state.protocol_state = self.policy.closed_state
            state.grants = []
        if selected == "clarify":
            state.clarifications += 1
        if state.decisions < self.policy.max_decisions:
            state.decisions += 1
            state.consumed_allowance += allowance
        state.revision += 1
        reasons = sorted(set(blocked.get(permit, [])))
        if selected != permit and not reasons:
            reasons = [f"policy_selected:{selected}"]
        decision = Decision(
            session_id=state.id,
            revision=state.revision,
            stage=candidate.stage,
            action=selected,
            allowed=selected == permit,
            message="Allowed by configured policy."
            if selected == permit
            else self.policy.actions[idx].message,
            reasons=reasons,
            action_probabilities=dict(zip(self.names, probs.tolist())),
            nominal_risks=dict(zip(self.names, nominal.tolist())),
            upper_risks=dict(zip(self.names, upper.tolist())),
            blocked_actions=blocked,
            belief=dict(zip(self.policy.states, state.belief)),
            protocol_state=state.protocol_state,
            candidate_digest=candidate.digest,
            policy_digest=self.policy.digest,
            risk_semantics=self.policy.risk_semantics,
            risk_allowance=allowance,
            consumed_allowance=state.consumed_allowance,
            observation=observation,
        )
        self._audit(state, candidate.stage, selected, candidate.digest, reasons)
        return decision
