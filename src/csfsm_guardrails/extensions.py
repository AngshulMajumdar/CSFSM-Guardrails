"""Small provider interfaces for application-specific evidence and checks."""

from __future__ import annotations

import re
import unicodedata
from typing import Protocol, Sequence

from .types import Candidate, Evidence, Finding, SessionState


class EvidenceEncoder(Protocol):
    def encode(self, text: str, state: SessionState) -> Evidence: ...


class CandidateValidator(Protocol):
    def validate(self, candidate: Candidate, state: SessionState) -> Sequence[Finding]: ...


class RiskProvider(Protocol):
    def risks(self, candidate: Candidate, state: SessionState) -> list[list[float]]:
        """Return an n_modes x n_actions conditional risk table or model scores."""
        ...


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    return "".join(c for c in text if unicodedata.category(c) != "Cf")


class RuleBasedEncoder:
    """Offline, transparent baseline. Scores are not calibrated probabilities.

    Replace this encoder and the observation table for a specific domain.
    It does not claim general jailbreak detection or universal language coverage.
    """

    probing = re.compile(
        r"(?is)\b(ignore|disregard|override|bypass)\b.{0,60}\b(instructions|guardrails|rules|policy|verification)\b|\b(reveal|print|dump)\b.{0,40}\b(system prompt|hidden instructions)\b"
    )
    restricted = re.compile(
        r"(?i)\b(steal credentials|dump customer records|another person's password|exfiltrate|stolen password)\b"
    )
    ambiguous = re.compile(r"(?i)^\s*(help|do it|this|that|continue|maybe|yes|no)[.!?\s]*$")

    def encode(self, text: str, state: SessionState) -> Evidence:
        text = normalize_text(text)
        if not text.strip():
            observation = "unknown"
        elif self.probing.search(text):
            observation = "probing"
        elif self.restricted.search(text):
            observation = "restricted"
        elif self.ambiguous.search(text):
            observation = "ambiguous"
        else:
            observation = "routine"
        return Evidence(
            observation=observation, source="rule_baseline", reasons=[f"evidence:{observation}"]
        )


class PatternValidator:
    def __init__(self, rules):
        self.rules = [(rule, re.compile(rule.pattern)) for rule in rules]

    def validate(self, candidate: Candidate, state: SessionState) -> list[Finding]:
        from .types import canonical_json

        text = candidate.text if candidate.stage != "tool" else canonical_json(candidate.arguments)
        text = normalize_text(text)
        return [
            Finding(code=rule.code)
            for rule, pattern in self.rules
            if candidate.stage in rule.stages and pattern.search(text)
        ]
