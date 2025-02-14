import pytest
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from model_metrics.model_calculator import ModelCalculator
import shap


@pytest.fixture
def sample_data():
    """Fixture to generate sample training and test data."""
    np.random.seed(42)
    X = pd.DataFrame(
        np.random.rand(100, 5),
        columns=[f"feature_{i}" for i in range(5)],
    )
    y = np.random.randint(0, 2, size=100)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


@pytest.fixture
def trained_models(sample_data):
    """Fixture to train models before testing."""
    X_train, X_test, y_train, y_test = sample_data

    log_reg = LogisticRegression()
    dt = DecisionTreeClassifier()
    rf = RandomForestClassifier()

    log_reg.fit(X_train, y_train)
    dt.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    return {
        "model": {
            "outcome_1": log_reg,
            "outcome_2": dt,
        }
    }


@pytest.fixture
def model_calculator(trained_models):
    """Fixture to create a ModelCalculator instance."""
    outcomes = ["outcome_1", "outcome_2"]
    return ModelCalculator(trained_models, outcomes, top_n=3)


def test_generate_predictions(model_calculator, sample_data):
    """Test model predictions and ensure output format is correct."""
    _, X_test, _, y_test = sample_data
    y_test_df = pd.DataFrame({"outcome_1": y_test, "outcome_2": y_test})

    result_df = model_calculator.generate_predictions(X_test, y_test_df)

    assert isinstance(result_df, pd.DataFrame)
    assert "y_pred_proba" in result_df.columns
    assert "TP" in result_df.columns
    assert "FP" in result_df.columns
    assert len(result_df) == len(X_test) * len(model_calculator.outcomes)


def test_generate_predictions_with_shap(model_calculator, sample_data):
    """Test generating predictions with SHAP values."""
    _, X_test, _, y_test = sample_data
    y_test_df = pd.DataFrame({"outcome_1": y_test, "outcome_2": y_test})

    result_df = model_calculator.generate_predictions(
        X_test, y_test_df, calculate_shap=True
    )

    assert isinstance(result_df, pd.DataFrame)
    assert any(col.startswith("top_") for col in result_df.columns)


def test_generate_predictions_with_coefficients(model_calculator, sample_data):
    """Test generating predictions with model coefficients."""
    _, X_test, _, y_test = sample_data
    y_test_df = pd.DataFrame({"outcome_1": y_test})

    model_calculator.outcomes = ["outcome_1"]

    result_df = model_calculator.generate_predictions(
        X_test, y_test_df, use_coefficients=True
    )

    assert isinstance(result_df, pd.DataFrame)
    assert any(col.startswith("top_") for col in result_df.columns)


def test_generate_predictions_with_conflicting_params(model_calculator, sample_data):
    """Test for ValueError when conflicting parameters are set."""
    _, X_test, _, y_test = sample_data
    y_test_df = pd.DataFrame({"outcome_1": y_test, "outcome_2": y_test})

    with pytest.raises(
        ValueError,
        match="Both 'calculate_shap' and 'use_coefficients' cannot be True simultaneously.",
    ):
        model_calculator.generate_predictions(
            X_test, y_test_df, calculate_shap=True, use_coefficients=True
        )

    with pytest.raises(
        ValueError,
        match="Both 'global_shap' and 'global_coefficients' cannot be True simultaneously.",
    ):
        model_calculator.generate_predictions(
            X_test, y_test_df, global_shap=True, global_coefficients=True
        )


def test_global_shap_values(model_calculator, sample_data):
    """Test computing global SHAP values."""
    _, X_test, _, y_test = sample_data
    y_test_df = pd.DataFrame({"outcome_1": y_test, "outcome_2": y_test})

    shap_df = model_calculator.generate_predictions(
        X_test,
        y_test_df,
        global_shap=True,
    )

    assert isinstance(shap_df, pd.DataFrame)
    assert "Feature" in shap_df.columns
    assert "SHAP Value" in shap_df.columns
    assert len(shap_df) > 0


def test_global_coefficients(model_calculator, sample_data):
    """Test computing global model coefficients."""
    _, X_test, _, y_test = sample_data
    y_test_df = pd.DataFrame({"outcome_1": y_test, "outcome_2": y_test})

    coeff_df = model_calculator.generate_predictions(
        X_test, y_test_df, global_coefficients=True
    )

    assert isinstance(coeff_df, pd.DataFrame)
    assert "Feature" in coeff_df.columns
    assert "Coefficient" in coeff_df.columns
    assert len(coeff_df) > 0


def test_extract_final_model(model_calculator, trained_models):
    """Test final model extraction from pipelines or models."""
    model = trained_models["model"]["outcome_1"]
    extracted_model = model_calculator._extract_final_model(model)

    assert extracted_model is not None
    assert hasattr(extracted_model, "predict")


def test_calculate_shap_values(model_calculator, sample_data):
    """Test SHAP value calculation."""
    _, X_test, _, _ = sample_data
    model = model_calculator.model_dict["model"]["outcome_1"]

    shap_values = model_calculator._calculate_shap_values(model, X_test)

    assert isinstance(shap_values, list)
    assert all(isinstance(row, list) for row in shap_values)
    assert len(shap_values) == len(X_test)


def test_calculate_coefficients(model_calculator, sample_data):
    """Test model coefficient calculation."""
    _, X_test, _, _ = sample_data
    model = model_calculator.model_dict["model"]["outcome_1"]

    coeff_values = model_calculator._calculate_coefficients(model, X_test)

    assert isinstance(coeff_values, list)
    assert all(isinstance(row, dict) for row in coeff_values)


def test_add_metrics(model_calculator, sample_data):
    """Test adding classification metrics to the test set."""
    _, X_test, _, y_test = sample_data
    y_pred = np.random.randint(0, 2, size=len(y_test))
    y_pred_proba = np.random.rand(len(y_test))

    result_df = model_calculator._add_metrics(
        X_test,
        y_test,
        y_pred,
        y_pred_proba,
        "outcome_1",
    )

    assert isinstance(result_df, pd.DataFrame)
    assert "TP" in result_df.columns
    assert "FN" in result_df.columns
    assert "FP" in result_df.columns
    assert "TN" in result_df.columns
    assert "y_pred_proba" in result_df.columns
