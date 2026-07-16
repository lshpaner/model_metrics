import numpy as np
import pandas as pd
import pytest
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor

from model_metrics.model_evaluator import show_feature_pareto


@pytest.fixture(autouse=True)
def _mpl_hygiene():
    orig_show = plt.show
    plt.show = lambda *a, **k: None
    try:
        yield
    finally:
        plt.close("all")
        plt.show = orig_show


@pytest.fixture(scope="module")
def rf():
    X, y = make_regression(
        n_samples=300, n_features=8, n_informative=5, noise=8, random_state=0
    )
    cols = ["bmi", "s5", "bp", "s6", "age", "s1", "s2", "s3"]
    X = pd.DataFrame(X, columns=cols)
    return RandomForestRegressor(n_estimators=50, random_state=0).fit(X, y)


def test_pareto_requires_model():
    with pytest.raises(ValueError):
        show_feature_pareto(None)


def test_pareto_bars_runs(rf):
    assert show_feature_pareto(rf, display="bars") is None


def test_pareto_curve_runs(rf):
    assert show_feature_pareto(rf, display="curve") is None


def test_pareto_invalid_display_raises(rf):
    with pytest.raises(ValueError):
        show_feature_pareto(rf, display="nope")


def test_pareto_return_df(rf):
    df = show_feature_pareto(rf, return_df=True)
    assert list(df.columns) == [
        "feature",
        "importance",
        "importance_pct",
        "cumulative_pct",
    ]


def test_pareto_return_features_is_list(rf):
    feats = show_feature_pareto(rf, return_features=True)
    assert isinstance(feats, list) and len(feats) >= 1


def test_pareto_return_both_is_dict(rf):
    out = show_feature_pareto(rf, return_features=True, return_df=True)
    assert set(out.keys()) == {"features", "df"}


def test_pareto_top_n_limits_bars(rf):
    # top_n truncates the drawn set but the returned df is still full
    df = show_feature_pareto(rf, top_n=3, return_df=True)
    assert len(df) == 8


def test_pareto_clean_names(rf):
    show_feature_pareto(rf, clean_names={"bmi": "Body Mass Index"})


def test_pareto_smooth_off(rf):
    show_feature_pareto(rf, smooth=False)


def test_pareto_saves_pdf(tmp_path, rf):
    show_feature_pareto(rf, image_filename=str(tmp_path / "pareto.pdf"))
    assert (tmp_path / "pareto.pdf").exists()


def test_pareto_accepts_ax(rf):
    fig, ax = plt.subplots()
    show_feature_pareto(rf, ax=ax)
