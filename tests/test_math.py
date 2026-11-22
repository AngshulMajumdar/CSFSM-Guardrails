import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from csfsm_guardrails.errors import ImpossibleObservation
from csfsm_guardrails.math import forward, forward_joint, masked_softmax, project, robust_risk


@given(
    st.lists(st.floats(min_value=1e-6, max_value=100, allow_nan=False), min_size=4, max_size=4),
    st.lists(st.floats(min_value=1e-6, max_value=100, allow_nan=False), min_size=4, max_size=4),
)
@settings(max_examples=100, deadline=None)
def test_forward_matches_joint(b, ell):
    b = np.array(b) / np.sum(b)
    transition = np.array(
        [[0.7, 0.1, 0.1, 0.1], [0.2, 0.5, 0.2, 0.1], [0.1, 0.2, 0.6, 0.1], [0.1, 0.1, 0.1, 0.7]]
    )
    value = forward(b, transition, ell)
    expected = b @ transition * ell
    expected /= expected.sum()
    assert np.allclose(value, expected)
    assert np.allclose(value, forward_joint(b, transition * ell))
    assert np.isclose(value.sum(), 1)


@given(
    st.lists(st.floats(min_value=-1000, max_value=1000, allow_nan=False), min_size=7, max_size=7),
    st.lists(st.booleans(), min_size=6, max_size=6),
)
@settings(max_examples=100, deadline=None)
def test_masked_projection_never_assigns_forbidden_probability(u, mask):
    mask = np.array(mask + [True])
    p = masked_softmax(u, mask, 0.7)
    assert np.isfinite(p).all()
    assert np.all(p[~mask] == 0)
    assert p.sum() == pytest.approx(1)


def test_projection_is_reference_conditioned_on_safe_support():
    assert np.allclose(project([0.2, 0.3, 0.5], [True, False, True]), [2 / 7, 0, 5 / 7])


@given(
    st.floats(min_value=0, max_value=1),
    st.floats(min_value=0, max_value=1),
    st.floats(min_value=0, max_value=1),
    st.floats(min_value=0, max_value=1),
)
@settings(max_examples=100, deadline=None)
def test_tv_bound_dominates_all_two_state_perturbations(b, r0, r1, eta):
    nominal, upper = robust_risk([b, 1 - b], [[r0], [r1]], eta)
    extreme = max(p * r0 + (1 - p) * r1 for p in [max(0, b - eta), min(1, b + eta)])
    assert upper[0] + 1e-12 >= extreme
    assert upper[0] <= max(r0, r1) + 1e-12
    assert nominal[0] <= upper[0] + 1e-12


def test_impossible_evidence_and_extreme_scale():
    with pytest.raises(ImpossibleObservation):
        forward([0.5, 0.5], np.eye(2), [0, 0])
    assert np.allclose(forward([0.5, 0.5], np.eye(2), [1e-300, 2e-300]), [1 / 3, 2 / 3])


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0])
def test_invalid_risk_rejected(bad):
    with pytest.raises(ValueError):
        robust_risk([0.5, 0.5], [[bad], [0.1]], 0.02)
