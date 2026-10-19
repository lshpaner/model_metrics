import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize

from model_metrics.model_evaluator import summarize_model_performance
from model_metrics.metrics_utils import compute_multiclass_metrics


# Move these two fixtures into conftest.py
@pytest.fixture
def multiclass_data():
    X, y = make_classification(
        n_samples=600, n_classes=3, n_informative=6, random_state=0
    )
    X = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    return X, pd.Series(y)


@pytest.fixture
def multiclass_model(multiclass_data):
    X, y = multiclass_data
    return LogisticRegression(max_iter=1000).fit(X, y)


def _metric(df, name, col="Model 1"):
    return df.loc[df["Metrics"] == name, col].iloc[0]


def test_auc_matches_sklearn_ovr(multiclass_data, multiclass_model):
    X, y = multiclass_data
    p = multiclass_model.predict_proba(X)
    df = summarize_model_performance(y_prob=p, y=y, return_df=True)
    expected = round(roc_auc_score(y, p, multi_class="ovr", average="macro"), 3)
    assert _metric(df, "AUC ROC") == expected


def test_brier_matches_hand_computation(multiclass_data, multiclass_model):
    X, y = multiclass_data
    p = multiclass_model.predict_proba(X)
    onehot = label_binarize(y, classes=[0, 1, 2])
    expected = round(np.mean(np.sum((p - onehot) ** 2, axis=1)), 3)
    df = summarize_model_performance(y_prob=p, y=y, return_df=True)
    assert _metric(df, "Brier Score") == expected


def test_model_path_matches_y_prob_path(multiclass_data, multiclass_model):
    X, y = multiclass_data
    a = summarize_model_performance(model=multiclass_model, X=X, y=y, return_df=True)
    b = summarize_model_performance(
        y_prob=multiclass_model.predict_proba(X), y=y, return_df=True
    )
    assert a.equals(b)


def test_string_labels(multiclass_data):
    X, y = multiclass_data
    ys = pd.Series(np.array(["a", "b", "c"])[y])
    m = LogisticRegression(max_iter=1000).fit(X, ys)
    df = summarize_model_performance(model=m, X=X, y=ys, return_df=True)
    assert 0.5 < _metric(df, "AUC ROC") <= 1.0


def test_no_threshold_row(multiclass_data, multiclass_model):
    X, y = multiclass_data
    df = summarize_model_performance(
        y_prob=multiclass_model.predict_proba(X), y=y, return_df=True
    )
    assert "Model Threshold" not in df["Metrics"].tolist()


@pytest.mark.parametrize(
    "kwargs",
    [{"custom_threshold": 0.4}, {"model_threshold": {"Model 1": 0.4}}],
)
def test_threshold_raises(multiclass_data, multiclass_model, kwargs):
    X, y = multiclass_data
    with pytest.raises(ValueError, match="Thresholds do not apply"):
        summarize_model_performance(
            y_prob=multiclass_model.predict_proba(X), y=y, **kwargs
        )


def test_invalid_average_raises(multiclass_data, multiclass_model):
    X, y = multiclass_data
    with pytest.raises(ValueError, match="average must be"):
        summarize_model_performance(
            y_prob=multiclass_model.predict_proba(X), y=y, average="samples"
        )


def test_grouped_skips_group_missing_a_class(multiclass_data, multiclass_model):
    X, y = multiclass_data
    rng = np.random.RandomState(0)
    groups = pd.Series(rng.choice(["F", "M"], len(y)))
    # a group of class-2 rows only, while F and M keep some class-2 rows
    groups[(y == 2) & (np.arange(len(y)) % 2 == 0)] = "only2"
    df = summarize_model_performance(
        y_prob=multiclass_model.predict_proba(X),
        y=y,
        group_category=groups,
        return_df=True,
    )
    assert set(df.columns) == {"Metrics", "F", "M"}


def test_micro_specificity_from_pooled_counts():
    y = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 0])
    p = np.eye(3)[y_pred] * 0.8 + 0.2 / 3
    out = compute_multiclass_metrics(y, y_pred, p, [0, 1, 2], average="micro")
    # 4 TP, 2 FP, 2 FN pooled; pooled TN = 3 classes * 6 rows - 4 - 2 - 2 = 10
    assert out["Specificity"] == round(10 / 12, 3)


def test_missing_class_raises():
    y = np.array([0, 0, 1, 1])
    p = np.full((4, 3), 1 / 3)
    with pytest.raises(ValueError, match="missing class"):
        compute_multiclass_metrics(y, p.argmax(1), p, [0, 1, 2])


def test_per_class_matches_one_vs_rest(multiclass_data, multiclass_model):
    X, y = multiclass_data
    p = multiclass_model.predict_proba(X)
    df = summarize_model_performance(y_prob=p, y=y, per_class=True, return_df=True)
    for j, c in enumerate(multiclass_model.classes_):
        auc = df.loc[df["Metrics"] == "AUC ROC", str(c)].iloc[0]
        assert auc == round(roc_auc_score(y == c, p[:, j]), 3)
        support = df.loc[df["Metrics"] == "Support", str(c)].iloc[0]
        assert support == int((y == c).sum())


def test_per_class_multiple_models_multiindex(multiclass_data, multiclass_model):
    X, y = multiclass_data
    p = multiclass_model.predict_proba(X)
    df = summarize_model_performance(
        y_prob=[p, p], y=y, model_title=["A", "B"], per_class=True, return_df=True
    )
    assert ("A", "0") in df.columns and ("B", "2") in df.columns


def test_per_class_binary_raises():
    X, y = make_classification(n_samples=200, random_state=0)
    p = LogisticRegression(max_iter=1000).fit(X, y).predict_proba(X)[:, 1]
    with pytest.raises(ValueError, match="multiclass models only"):
        summarize_model_performance(y_prob=p, y=y, per_class=True)


def test_per_class_with_groups_raises(multiclass_data, multiclass_model):
    X, y = multiclass_data
    with pytest.raises(ValueError, match="group_category"):
        summarize_model_performance(
            y_prob=multiclass_model.predict_proba(X),
            y=y,
            per_class=True,
            group_category=pd.Series(["a"] * len(y)),
        )


def test_split_by_model_returns_dict(multiclass_data, multiclass_model):
    X, y = multiclass_data
    p = multiclass_model.predict_proba(X)
    out = summarize_model_performance(
        y_prob=[p, p],
        y=y,
        model_title=["A", "B"],
        per_class=True,
        split_by_model=True,
        return_df=True,
    )
    assert list(out) == ["A", "B"]
    single = summarize_model_performance(
        y_prob=p, y=y, model_title="A", per_class=True, return_df=True
    )
    assert out["A"].equals(single)


def test_split_by_model_requires_per_class(multiclass_data, multiclass_model):
    X, y = multiclass_data
    with pytest.raises(ValueError, match="requires per_class"):
        summarize_model_performance(
            y_prob=multiclass_model.predict_proba(X), y=y, split_by_model=True
        )

