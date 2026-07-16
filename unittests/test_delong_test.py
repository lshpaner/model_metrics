import numpy as np
import pytest

from model_metrics.delong_test import delong_roc_test, hanley_mcneil_auc_test
from sklearn.metrics import roc_auc_score


@pytest.fixture
def data():
    rng = np.random.RandomState(0)
    y = rng.binomial(1, 0.4, 500)
    s1 = y * 0.8 + rng.randn(500)
    s2 = y * 1.1 + rng.randn(500)  # s2 clearly better
    return y, s1, s2


# --- delong_roc_test -----------------------------------------------------------------
def test_delong_returns_triplet(data):
    y, s1, s2 = data
    out = delong_roc_test(y, s1, s2, return_values=True, verbose=False)
    assert isinstance(out, tuple) and len(out) == 3


def test_delong_aucs_match_sklearn(data):
    y, s1, s2 = data
    a1, a2, _ = delong_roc_test(y, s1, s2, return_values=True, verbose=False)
    assert np.isclose(a1, roc_auc_score(y, s1))
    assert np.isclose(a2, roc_auc_score(y, s2))


def test_delong_identical_scores_p_is_one(data):
    y, s1, _ = data
    _, _, p = delong_roc_test(y, s1, s1, return_values=True, verbose=False)
    assert p == pytest.approx(1.0)


def test_delong_symmetric(data):
    y, s1, s2 = data
    p12 = delong_roc_test(y, s1, s2, return_values=True, verbose=False)[2]
    p21 = delong_roc_test(y, s2, s1, return_values=True, verbose=False)[2]
    assert p12 == pytest.approx(p21)


def test_delong_detects_clear_difference(data):
    y, s1, s2 = data
    _, _, p = delong_roc_test(y, s1, s2, return_values=True, verbose=False)
    assert 0.0 <= p <= 1.0


def test_delong_requires_binary():
    with pytest.raises(ValueError):
        delong_roc_test([0, 1, 2, 1], [0.1, 0.2, 0.3, 0.4], [0.2, 0.1, 0.4, 0.3], verbose=False)


def test_delong_requires_both_classes():
    with pytest.raises(ValueError):
        delong_roc_test([1, 1, 1], [0.1, 0.2, 0.3], [0.2, 0.1, 0.3], verbose=False)


def test_delong_handles_ties(data):
    y, _, _ = data
    tied = np.random.RandomState(1).randint(0, 4, len(y)).astype(float)
    _, _, p = delong_roc_test(y, tied, tied, return_values=True, verbose=False)
    assert p == pytest.approx(1.0)


def test_delong_verbose_prints(data, capsys):
    y, s1, s2 = data
    delong_roc_test(y, s1, s2, verbose=True)
    assert "DeLong" in capsys.readouterr().out


# --- hanley_mcneil_auc_test (approximation) ------------------------------------------
def test_hanley_returns_triplet(data):
    y, s1, s2 = data
    out = hanley_mcneil_auc_test(y, s1, s2, return_values=True, verbose=False)
    assert isinstance(out, tuple) and len(out) == 3


def test_hanley_identical_scores_p_is_one(data):
    y, s1, _ = data
    _, _, p = hanley_mcneil_auc_test(y, s1, s1, return_values=True, verbose=False)
    assert p == pytest.approx(1.0)


def test_hanley_verbose_prints(data, capsys):
    y, s1, s2 = data
    hanley_mcneil_auc_test(y, s1, s2, verbose=True)
    assert "Hanley" in capsys.readouterr().out
