import pytest
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from model_metrics.model_evaluator import (
    save_plot_images,
    get_predictions,
    summarize_model_performance,
    show_confusion_matrix,
    get_model_probabilities,
    extract_model_titles,
    extract_model_name,
)


@pytest.fixture
def sample_data():
    """Fixture to create a sample dataset for testing."""
    np.random.seed(42)
    X = pd.DataFrame(np.random.rand(100, 3), columns=["A", "B", "C"])
    y = np.random.randint(0, 2, size=100)
    return X, y


@pytest.fixture
def trained_model(sample_data):
    """Fixture to train a sample model for testing."""
    X, y = sample_data
    model = LogisticRegression()
    model.fit(X, y)
    return model


def test_save_plot_images(tmp_path):
    """Test that images are saved correctly when save_plot=True."""
    filename = "test_plot"
    image_path_png = tmp_path / "png"
    image_path_svg = tmp_path / "svg"

    os.makedirs(image_path_png, exist_ok=True)
    os.makedirs(image_path_svg, exist_ok=True)

    plt.plot([0, 1], [0, 1])  # Dummy plot
    save_plot_images(filename, True, str(image_path_png), str(image_path_svg))

    assert os.path.exists(os.path.join(image_path_png, f"{filename}.png"))
    assert os.path.exists(os.path.join(image_path_svg, f"{filename}.svg"))


def test_get_predictions(trained_model, sample_data):
    """Test get_predictions function."""
    X, y = sample_data
    y_true, y_prob, y_pred, threshold = get_predictions(
        trained_model, X, y, None, None, None
    )

    assert len(y_true) == len(y_prob) == len(y_pred) == len(y)
    assert 0 <= threshold <= 1


def test_summarize_model_performance(trained_model, sample_data):
    """Test summarize_model_performance function."""
    X, y = sample_data
    df = summarize_model_performance([trained_model], X, y, return_df=True)

    assert isinstance(df, pd.DataFrame)
    assert "Precision/PPV" in df.index
    assert "AUC ROC" in df.index


def test_get_model_probabilities(trained_model, sample_data):
    """Test extracting model probabilities."""
    X, _ = sample_data
    probs = get_model_probabilities(trained_model, X, "TestModel")

    assert len(probs) == len(X)
    assert np.all(0 <= probs) and np.all(
        probs <= 1
    )  # Probabilities should be between 0 and 1


def test_extract_model_titles(trained_model):
    """Test extracting model titles."""
    titles = extract_model_titles([trained_model], model_titles=["LogisticRegression"])
    assert titles == ["LogisticRegression"]


def test_extract_model_name(trained_model):
    """Test extracting model names."""
    name = extract_model_name(trained_model)
    assert name == "LogisticRegression"
