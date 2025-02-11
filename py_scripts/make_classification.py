import pandas as pd
import numpy as np

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from model_tuner import Model


# Generate a synthetic dataset
X, y = make_classification(
    n_samples=10000,  # Number of samples
    n_features=50,  # Number of features
    n_informative=25,  # Number of informative features
    n_classes=2,  # Number of classes (binary classification)
    random_state=42,  # Reproducibility
)

# Convert to a DataFrame
df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(50)])
df["target"] = y

# Convert y into a DataFrame
y = pd.DataFrame(y, columns=["target"]).squeeze()

# Convert X into a DataFrame
X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])

lr = LogisticRegression(class_weight="balanced", max_iter=1000)

estimator_name = "lg"
# Set the parameters by cross-validation
tuned_parameters = [
    {
        estimator_name + "__C": np.logspace(-4, 0, 3),
    }
]
kfold = False
calibrate = False


model = Model(
    name="Logistic Regression",
    estimator_name=estimator_name,
    model_type="classification",
    calibrate=calibrate,
    estimator=lr,
    kfold=kfold,
    stratify_y=False,
    grid=tuned_parameters,
    randomized_grid=False,
    scoring=["roc_auc"],
    n_splits=10,
    n_jobs=-2,
    random_state=42,
)


model.grid_search_param_tuning(X, y)

X_train, y_train = model.get_train_data(X, y)
X_test, y_test = model.get_test_data(X, y)
X_valid, y_valid = model.get_valid_data(X, y)

model.fit(X_train, y_train)

print("Validation Metrics")
model.return_metrics(
    X_valid,
    y_valid,
    optimal_threshold=True,
    print_threshold=True,
    model_metrics=True,
)
print()
print("Test Metrics")
model.return_metrics(
    X_test,
    y_test,
    optimal_threshold=True,
    print_threshold=True,
    model_metrics=True,
)

y_prob = model.predict_proba(X_test)

### F1 Weighted
y_pred = model.predict(X_test, optimal_threshold=True)

### Report Model Metrics

model.return_metrics(
    X_test,
    y_test,
    optimal_threshold=True,
    print_threshold=True,
    model_metrics=True,
    return_dict=False,
)
