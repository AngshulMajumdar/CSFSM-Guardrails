"""Pure numerical primitives corresponding to the mathematical specification."""

import numpy as np

from .errors import ImpossibleObservation


def normalize(v):
    v = np.asarray(v, dtype=float)
    if v.ndim != 1 or not np.isfinite(v).all() or (v < 0).any():
        raise ValueError("Expected a finite nonnegative vector")
    scale = v.max(initial=0)
    if scale <= 0:
        raise ImpossibleObservation("Observation has zero model likelihood")
    v = v / scale
    return v / v.sum()


def forward(belief, transition, likelihood):
    b, t, ell = np.asarray(belief), np.asarray(transition), np.asarray(likelihood)
    if t.shape != (b.size, b.size) or ell.shape != b.shape:
        raise ValueError("Inconsistent filter dimensions")
    if not np.isfinite(t).all() or (t < 0).any() or not np.isfinite(ell).all() or (ell < 0).any():
        raise ValueError("Invalid transition or likelihood")
    # Scale likelihood before multiplying to avoid unnecessary underflow.
    if ell.max(initial=0) <= 0:
        raise ImpossibleObservation("Zero likelihood")
    return normalize((normalize(b) @ t) * (ell / ell.max()))


def forward_joint(belief, matrix):
    m = np.asarray(matrix, dtype=float)
    b = normalize(belief)
    if m.shape != (b.size, b.size) or not np.isfinite(m).all() or (m < 0).any():
        raise ValueError("Invalid joint observation matrix")
    scale = m.max(initial=0)
    if scale <= 0:
        raise ImpossibleObservation("Missing or impossible joint observation")
    return normalize(b @ (m / scale))


def robust_risk(belief, risks, radius=0.0):
    b, r = normalize(belief), np.asarray(risks, dtype=float)
    if (
        r.ndim != 2
        or r.shape[0] != b.size
        or not np.isfinite(r).all()
        or (r < 0).any()
        or (r > 1).any()
    ):
        raise ValueError("Risk matrix must contain finite probabilities")
    if not np.isfinite(radius) or not 0 <= radius <= 1:
        raise ValueError("TV radius must be in [0, 1]")
    nominal = b @ r
    upper = np.minimum(r.max(axis=0), nominal + radius * np.ptp(r, axis=0))
    return nominal, upper


def project(reference, allowed):
    p, mask = np.asarray(reference, dtype=float), np.asarray(allowed, dtype=bool)
    if (
        p.shape != mask.shape
        or p.ndim != 1
        or not mask.any()
        or not np.isfinite(p).all()
        or (p < 0).any()
    ):
        raise ValueError("Invalid reference or empty admissible support")
    return normalize(np.where(mask, p, 0.0))


def masked_softmax(utilities, allowed, temperature=1.0):
    u, mask = np.asarray(utilities, dtype=float), np.asarray(allowed, dtype=bool)
    if (
        u.ndim != 1
        or u.shape != mask.shape
        or not mask.any()
        or not np.isfinite(u).all()
        or not np.isfinite(temperature)
        or temperature <= 0
    ):
        raise ValueError("Invalid utilities, temperature or empty support")
    out = np.zeros_like(u)
    # Subtract before division; forbidden entries never enter exp or normalization.
    out[mask] = np.exp((u[mask] - np.max(u[mask])) / temperature)
    out[mask] /= out[mask].sum()
    return out


def posterior_to_likelihood(posterior, training_prior):
    p, prior = normalize(posterior), normalize(training_prior)
    if p.shape != prior.shape or (prior <= 0).any():
        raise ValueError("A positive, matching training prior is required")
    return normalize(p / prior)
