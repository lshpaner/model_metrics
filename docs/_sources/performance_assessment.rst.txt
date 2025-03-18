.. _performance_assessment:   

.. Model Metrics documentation master file, created by
   sphinx-quickstart on Sun Feb 16 11:29:04 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. _target-link:

.. raw:: html

   <div class="no-click">

.. image:: ../assets/mm_logo.svg
   :alt: Model Metrics Logo
   :align: left
   :width: 300px

.. raw:: html

   </div>

.. raw:: html
   
   <div style="height: 100px;"></div>

Model Performance Summaries
-----------------------------

**Summarizes model performance metrics for classification and regression models.**

.. function:: summarize_model_performance(model, X, y, model_type="classification", model_threshold=None, model_title=None, custom_threshold=None, score=None, return_df=False, overall_only=False, decimal_places=3)

    :param model: A trained model or a list of trained models.
    :type model: object or list
    :param X: Feature matrix used for evaluation.
    :type X: pd.DataFrame
    :param y: Target variable.
    :type y: pd.Series or np.array
    :param model_type: Specifies whether the model is for classification or regression. Must be either ``"classification"`` or ``"regression"``. Defaults to ``"classification"``.
    :type model_type: str, optional
    :param model_threshold: Threshold values for classification models. If provided, this dictionary specifies thresholds per model. Defaults to ``None``.
    :type model_threshold: dict, optional
    :param model_title: Custom model names for display. If ``None``, names are inferred from the models. Defaults to ``None``.
    :type model_title: str or list, optional
    :param custom_threshold: A fixed threshold for classification, overriding ``model_threshold``. If set, the `"Model Threshold"` row is excluded. Defaults to ``None``.
    :type custom_threshold: float, optional
    :param score: A custom scoring metric for classification models. Defaults to ``None``.
    :type score: str, optional
    :param return_df: If ``True``, returns a DataFrame instead of printing results. Defaults to ``False``.
    :type return_df: bool, optional
    :param overall_only: If ``True``, returns only the `"Overall Metrics"` row, removing coefficient-related columns for regression models. Defaults to ``False``.
    :type overall_only: bool, optional
    :param decimal_places: Number of decimal places to round numerical metrics. Defaults to ``3``.
    :type decimal_places: int, optional

    :returns: A DataFrame containing model performance metrics if ``return_df=True``. Otherwise, the metrics are printed in a formatted table.
    :rtype: pd.DataFrame or None

    :raises ValueError: If ``model_type="classification"`` and ``overall_only=True``.
    :raises ValueError: If ``model_type`` is not ``"classification"`` or ``"regression"``.

.. admonition:: Notes

    - **Classification Models:**
        - Computes precision, recall, specificity, AUC ROC, F1-score, Brier score, and other key metrics.
        - Requires models supporting ``predict_proba`` or ``decision_function``.
        - If ``custom_threshold`` is set, it overrides ``model_threshold``.
    
    - **Regression Models:**
        - Computes MAE, MAPE, MSE, RMSE, Explained Variance, and R² Score.
        - Uses ``statsmodels.OLS`` to extract coefficients and p-values.
        - If ``overall_only=True``, the DataFrame retains only overall performance metrics.

    - **All Models:**
        - When ``decimal_places`` is specified with a desired number, it controls the precision of decimal places displayed in the results.
        - If ``return_df=False``, the function outputs results in a printed, formatted, readable structure instead of returning a DataFrame.

The ``summarize_model_performance`` function provides a structured evaluation of classification and regression models, generating key performance metrics. For classification models, it computes precision, recall, specificity, F1-score, and AUC ROC. For regression models, it extracts coefficients and evaluates error metrics like MSE, RMSE, and R². The function allows specifying custom thresholds, metric rounding, and formatted display options.

Below are two examples demonstrating how to evaluate multiple models using ``summarize_model_performance``. The function calculates and presents metrics for classification and regression models.

.. _Binary_Classification:

Binary Classification Models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This section introduces binary classification using two widely used machine 
learning models: Logistic Regression and Random Forest Classifier.

These examples demonstrate how ``model_metrics`` prepares and trains models on a 
synthetic dataset, setting the stage for evaluating their performance in subsequent sections. 
Both models use a default classification threshold of 0.5, where predictions are 
classified as positive (1) if the predicted probability exceeds 0.5, and negative (0) 
otherwise.


.. code-block:: python

    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    # Generate a synthetic dataset
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        random_state=42,
    )

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    # Train models
    model1 = LogisticRegression().fit(X_train, y_train)
    model2 = RandomForestClassifier().fit(X_train, y_train)

    model_title = ["Logistic Regression", "Random Forest"]



Binary Classification Example 1
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from model_metrics import summarize_model_performance

    model_performance = summarize_model_performance(
        model=[model1, model2],
        model_title=model_title,
        X=X_test,
        y=y_test,
        model_type="classification",
        return_df=True,
    )

    model_performance


**Output**

.. raw:: html

    <style type="text/css">
    .tg {
        border-collapse: collapse;
        border-spacing: 0;
        max-width: 450px; /* Fixed maximum width */
        width: 100%;
    }
    .tg td {
        border-color: black;
        border-style: solid;
        border-width: 1px;
        font-family: Arial, sans-serif;
        font-size: 12px; /* Reduced from 14px */
        overflow: hidden;
        padding: 0px 3px; /* Reduced from 0px 5px */
        word-break: normal;
    }
    .tg th {
        border-color: black;
        border-style: solid;
        border-width: 1px;
        font-family: Arial, sans-serif;
        font-size: 12px; /* Reduced from 14px */
        font-weight: normal;
        overflow: hidden;
        padding: 0px 3px; /* Reduced from 0px 5px */
        word-break: normal;
    }
    .tg .tg-kex3 { font-weight: bold; text-align: right; vertical-align: bottom; }
    .tg .tg-j6zm { font-weight: bold; text-align: right; vertical-align: bottom; }
    .tg .tg-2b7s { text-align: right; vertical-align: bottom; }
    .tg .tg-7zrl { text-align: right; vertical-align: bottom; }

    @media screen and (max-width: 767px) {
        .tg { width: auto !important; }
        .tg col { width: auto !important; }
        .tg-wrap { overflow-x: auto; -webkit-overflow-scrolling: touch; }
    }
    </style>

    <div class="tg-wrap">
        <table class="tg" style="table-layout: fixed; width: 450px;">
            <colgroup>
                <col style="width: 231px"> <!-- Metrics -->
                <col style="width: 231px"> <!-- Logistic Regression -->
                <col style="width: 231px"> <!-- Random Forest -->
            </colgroup>
            <thead>
                <tr>
                    <th class="tg-kex3">Metrics</th>
                    <th class="tg-j6zm">Logistic Regression</th>
                    <th class="tg-j6zm">Random Forest</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td class="tg-2b7s">Precision/PPV</td>
                    <td class="tg-7zrl">0.867</td>
                    <td class="tg-7zrl">0.912</td>
                </tr>
                <tr>
                    <td class="tg-2b7s">Average Precision</td>
                    <td class="tg-7zrl">0.937</td>
                    <td class="tg-7zrl">0.966</td>
                </tr>
                <tr>
                    <td class="tg-2b7s">Sensitivity/Recall</td>
                    <td class="tg-7zrl">0.82</td>
                    <td class="tg-7zrl">0.838</td>
                </tr>
                <tr>
                    <td class="tg-2b7s">Specificity</td>
                    <td class="tg-7zrl">0.843</td>
                    <td class="tg-7zrl">0.899</td>
                </tr>
                <tr>
                    <td class="tg-2b7s">F1-Score</td>
                    <td class="tg-7zrl">0.843</td>
                    <td class="tg-7zrl">0.873</td>
                </tr>
                <tr>
                    <td class="tg-2b7s">AUC ROC</td>
                    <td class="tg-7zrl">0.913</td>
                    <td class="tg-7zrl">0.95</td>
                </tr>
                <tr>
                    <td class="tg-2b7s">Brier Score</td>
                    <td class="tg-7zrl">0.118</td>
                    <td class="tg-7zrl">0.086</td>
                </tr>
                <tr>
                    <td class="tg-2b7s">Model Threshold</td>
                    <td class="tg-7zrl">0.5</td>
                    <td class="tg-7zrl">0.5</td>
                </tr>
            </tbody>
        </table>
    </div>

.. raw:: html

    <div style="height: 40px;"></div>


Binary Classification Example 2
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this example, we revisit binary classification with the same two models—Logistic 
Regression and Random Forest—but adjust the classification threshold 
(``custom_threshold`` input in this case) from the default 0.5 to 0.2. This 
change allows us to explore how lowering the threshold impacts model performance, 
potentially increasing sensitivity (recall) by classifying more instances as 
positive (1) at the expense of precision.

.. code-block:: python

    from model_metrics import summarize_model_performance

    model_performance = summarize_model_performance(
        model=[model1, model2],
        model_title=model_title,
        X=X_test,
        y=y_test,
        model_type="classification",
        return_df=True,
        custom_threshold=0.2,
    )

    model_performance


**Output**

.. raw:: html

    <style type="text/css">
    .tg {
        border-collapse: collapse;
        border-spacing: 0;
        max-width: 450px; /* Fixed maximum width */
        width: 100%;
    }
    .tg td {
        border-color: black;
        border-style: solid;
        border-width: 1px;
        font-family: Arial, sans-serif;
        font-size: 12px; /* Reduced from 14px */
        overflow: hidden;
        padding: 0px 3px; /* Reduced from 0px 5px */
        word-break: normal;
    }
    .tg th {
        border-color: black;
        border-style: solid;
        border-width: 1px;
        font-family: Arial, sans-serif;
        font-size: 12px; /* Reduced from 14px */
        font-weight: normal;
        overflow: hidden;
        padding: 0px 3px; /* Reduced from 0px 5px */
        word-break: normal;
    }
    .tg .tg-kex3 { font-weight: bold; text-align: right; vertical-align: bottom; }
    .tg .tg-j6zm { font-weight: bold; text-align: right; vertical-align: bottom; }
    .tg .tg-2b7s { text-align: right; vertical-align: bottom; }
    .tg .tg-7zrl { text-align: right; vertical-align: bottom; }

    @media screen and (max-width: 767px) {
        .tg { width: auto !important; }
        .tg col { width: auto !important; }
        .tg-wrap { overflow-x: auto; -webkit-overflow-scrolling: touch; }
    }
    </style>

    <div class="tg-wrap">
        <table class="tg" style="table-layout: fixed; width: 450px;">
            <colgroup>
                <col style="width: 231px"> <!-- Metrics -->
                <col style="width: 231px"> <!-- Logistic Regression -->
                <col style="width: 231px"> <!-- Random Forest -->
            </colgroup>
            <thead>
                <tr>
                    <th class="tg-kex3">Metrics</th>
                    <th class="tg-j6zm">Logistic Regression</th>
                    <th class="tg-j6zm">Random Forest</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td class="tg-2b7s">Precision/PPV</td>
                    <td class="tg-7zrl">0.803</td>
                    <td class="tg-7zrl">0.831</td>
                </tr>
                <tr>
                    <td class="tg-2b7s">Average Precision</td>
                    <td class="tg-7zrl">0.937</td>
                    <td class="tg-7zrl">0.966</td>
                </tr>
                <tr>
                    <td class="tg-2b7s">Sensitivity/Recall</td>
                    <td class="tg-7zrl">0.919</td>
                    <td class="tg-7zrl">0.928</td>
                </tr>
                <tr>
                    <td class="tg-2b7s">Specificity</td>
                    <td class="tg-7zrl">0.719</td>
                    <td class="tg-7zrl">0.764</td>
                </tr>
                <tr>
                    <td class="tg-2b7s">F1-Score</td>
                    <td class="tg-7zrl">0.857</td>
                    <td class="tg-7zrl">0.877</td>
                </tr>
                <tr>
                    <td class="tg-2b7s">AUC ROC</td>
                    <td class="tg-7zrl">0.913</td>
                    <td class="tg-7zrl">0.949</td>
                </tr>
                <tr>
                    <td class="tg-2b7s">Brier Score</td>
                    <td class="tg-7zrl">0.118</td>
                    <td class="tg-7zrl">0.085</td>
                </tr>
                <tr>
                    <td class="tg-2b7s">Model Threshold</td>
                    <td class="tg-7zrl">0.2</td>
                    <td class="tg-7zrl">0.2</td>
                </tr>
            </tbody>
        </table>
    </div>

.. raw:: html

    <div style="height: 40px;"></div>


Regression Models
^^^^^^^^^^^^^^^^^^^^

In this section, we load the `diabetes dataset` [1]_ from ``scikit-learn``, which includes 
features like age and BMI, along with a target variable representing disease 
progression. The data is then split with ``train_test_split`` into training and 
testing sets using an 80/20 ratio to facilitate model assessment. We train a 
Linear Regression model on unscaled data for a straightforward baseline, followed b
y a Random Forest Regressor with 100 trees, also on unscaled data, to introduce a 
more complex approach. Additionally, we train a Ridge Regression model using a 
``Pipeline`` that scales the features with ``StandardScaler`` before fitting, 
incorporating regularization. These steps prepare the models for subsequent 
evaluation and comparison using tools provided by the ``model_metrics`` library.

**Models use in these regression examples:**

- **Linear Regression:** A foundational model trained on unscaled data, simple yet effective for baseline evaluation.
- **Ridge Regression:** A regularized model with a Pipeline for scaling, perfect for testing stability and overfitting.
- **Random Forest Regressor:** An ensemble of 100 trees on unscaled data, offering complexity for comparative analysis.



.. code-block:: python

    import pandas as pd
    from sklearn.datasets import load_diabetes
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.pipeline import Pipeline

    # Load dataset
    diabetes = load_diabetes(as_frame=True)["frame"]
    X = diabetes.drop(columns=["target"])
    y = diabetes["target"]

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    # Train Linear Regression (on unscaled data)
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)

    # Train Random Forest Regressor (on unscaled data)
    rf_model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
    )
    rf_model.fit(X_train, y_train)

    # Train Ridge Regression (on scaled data)
    ridge_model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("estimator", Ridge(alpha=1.0)),
        ]
    )
    ridge_model.fit(X_train, y_train)



Regression Example 1
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from model_metrics import summarize_model_performance

    regression_metrics = summarize_model_performance(
        model=[linear_model, ridge_model],
        model_title=["Linear Regression", "Ridge Regression"],
        X=X_test,
        y=y_test,
        model_type="regression",
        return_df=True,
    )

    regression_metrics


The output below presents a detailed comparison of the performance and coefficients 
for two regression models—Linear Regression and Ridge Regression—trained on the 
diabetes dataset. It includes overall metrics such as Mean Absolute Error (MAE), 
Mean Absolute Percentage Error (MAPE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), 
Explained Variance, and R² Score for each model, showing their predictive accuracy. 
Additionally, it lists the coefficients for each feature (e.g., age, bmi, s1–s6) 
in both models, highlighting how each variable contributes to the prediction. 
This output serves as a foundation for evaluating and comparing the models’ 
effectiveness in [Your Library Name]'s documentation.

**Output**

.. raw:: html

    <style type="text/css">
    .tg {
        border-collapse: collapse;
        border-spacing: 0;
        max-width: 450px; /* Fixed maximum width */
        width: 100%;
    }
    .tg td {
        border-color: black;
        border-style: solid;
        border-width: 1px;
        font-family: Arial, sans-serif;
        font-size: 12px; /* Reduced from 14px */
        overflow: hidden;
        padding: 0px 3px; /* Reduced from 0px 5px */
        word-break: normal;
    }
    .tg th {
        border-color: black;
        border-style: solid;
        border-width: 1px;
        font-family: Arial, sans-serif;
        font-size: 12px; /* Reduced from 14px */
        font-weight: normal;
        overflow: hidden;
        padding: 0px 3px; /* Reduced from 0px 5px */
        word-break: normal;
    }
    .tg .tg-oxe6 { font-weight: bold; text-align: right; vertical-align: bottom; }
    .tg .tg-0thz { font-weight: bold; text-align: right; vertical-align: bottom; }
    .tg .tg-jkyp { text-align: right; vertical-align: bottom; }
    .tg .tg-za14 { text-align: right; vertical-align: bottom; }

    @media screen and (max-width: 767px) {
        .tg { width: auto !important; }
        .tg col { width: auto !important; }
        .tg-wrap { overflow-x: auto; -webkit-overflow-scrolling: touch; }
    }
    </style>

    <div class="tg-wrap">
        <table class="tg" style="table-layout: fixed; width: 450px;">
            <colgroup>
                <col style="width: 110px"> <!-- Model -->
                <col style="width: 90px"> <!-- Metric -->
                <col style="width: 60px"> <!-- Variable -->
                <col style="width: 70px"> <!-- Coefficient -->
                <col style="width: 50px"> <!-- MAE -->
                <col style="width: 50px"> <!-- MAPE -->
                <col style="width: 60px"> <!-- MSE -->
                <col style="width: 60px"> <!-- RMSE -->
                <col style="width: 70px"> <!-- Expl. Var. -->
                <col style="width: 70px"> <!-- R² Score -->
            </colgroup>
            <thead>
                <tr>
                    <th class="tg-oxe6">Model</th>
                    <th class="tg-oxe6">Metric</th>
                    <th class="tg-oxe6">Variable</th>
                    <th class="tg-oxe6">Coefficient</th>
                    <th class="tg-oxe6">MAE</th>
                    <th class="tg-oxe6">MAPE</th>
                    <th class="tg-oxe6">MSE</th>
                    <th class="tg-oxe6">RMSE</th>
                    <th class="tg-0thz">Expl. Var.</th>
                    <th class="tg-0thz">R^2 Score</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td class="tg-jkyp">Linear Regression</td>
                    <td class="tg-jkyp">Overall Metrics</td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp">42.794</td>
                    <td class="tg-jkyp">37.5</td>
                    <td class="tg-jkyp">2900.194</td>
                    <td class="tg-jkyp">53.853</td>
                    <td class="tg-za14">0.455</td>
                    <td class="tg-za14">0.453</td>
                </tr>
                <tr>
                    <td class="tg-jkyp">Linear Regression</td>
                    <td class="tg-jkyp">Coefficient</td>
                    <td class="tg-jkyp">const</td>
                    <td class="tg-jkyp">151.346</td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-za14"></td>
                    <td class="tg-za14"></td>
                </tr>
                <tr>
                    <td class="tg-jkyp">Linear Regression</td>
                    <td class="tg-jkyp">Coefficient</td>
                    <td class="tg-jkyp">age</td>
                    <td class="tg-jkyp">37.904</td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-za14"></td>
                    <td class="tg-za14"></td>
                </tr>
                <tr>
                    <td class="tg-jkyp">Linear Regression</td>
                    <td class="tg-jkyp">Coefficient</td>
                    <td class="tg-jkyp">sex</td>
                    <td class="tg-jkyp">-241.964</td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-za14"></td>
                    <td class="tg-za14"></td>
                </tr>
                <tr>
                    <td class="tg-jkyp">Linear Regression</td>
                    <td class="tg-jkyp">Coefficient</td>
                    <td class="tg-jkyp">bmi</td>
                    <td class="tg-jkyp">542.429</td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-za14"></td>
                    <td class="tg-za14"></td>
                </tr>
                <tr>
                    <td class="tg-jkyp">Linear Regression</td>
                    <td class="tg-jkyp">Coefficient</td>
                    <td class="tg-jkyp">bp</td>
                    <td class="tg-jkyp">347.704</td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-za14"></td>
                    <td class="tg-za14"></td>
                </tr>
                <tr>
                    <td class="tg-jkyp">Linear Regression</td>
                    <td class="tg-jkyp">Coefficient</td>
                    <td class="tg-jkyp">s1</td>
                    <td class="tg-jkyp">-931.489</td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-za14"></td>
                    <td class="tg-za14"></td>
                </tr>
                <tr>
                    <td class="tg-jkyp">Linear Regression</td>
                    <td class="tg-jkyp">Coefficient</td>
                    <td class="tg-jkyp">s2</td>
                    <td class="tg-jkyp">518.062</td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-za14"></td>
                    <td class="tg-za14"></td>
                </tr>
                <tr>
                    <td class="tg-jkyp">Linear Regression</td>
                    <td class="tg-jkyp">Coefficient</td>
                    <td class="tg-jkyp">s3</td>
                    <td class="tg-jkyp">163.42</td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-za14"></td>
                    <td class="tg-za14"></td>
                </tr>
                <tr>
                    <td class="tg-jkyp">Linear Regression</td>
                    <td class="tg-jkyp">Coefficient</td>
                    <td class="tg-jkyp">s4</td>
                    <td class="tg-jkyp">275.318</td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-za14"></td>
                    <td class="tg-za14"></td>
                </tr>
                <tr>
                    <td class="tg-jkyp">Linear Regression</td>
                    <td class="tg-jkyp">Coefficient</td>
                    <td class="tg-jkyp">s5</td>
                    <td class="tg-jkyp">736.199</td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-za14"></td>
                    <td class="tg-za14"></td>
                </tr>
                <tr>
                    <td class="tg-jkyp">Linear Regression</td>
                    <td class="tg-jkyp">Coefficient</td>
                    <td class="tg-jkyp">s6</td>
                    <td class="tg-jkyp">48.671</td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-za14"></td>
                    <td class="tg-za14"></td>
                </tr>
                <tr>
                    <td class="tg-jkyp">Ridge Regression</td>
                    <td class="tg-jkyp">Overall Metrics</td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp">42.812</td>
                    <td class="tg-jkyp">37.448</td>
                    <td class="tg-jkyp">2892.015</td>
                    <td class="tg-jkyp">53.777</td>
                    <td class="tg-za14">0.457</td>
                    <td class="tg-za14">0.454</td>
                </tr>
                <tr>
                    <td class="tg-jkyp">Ridge Regression</td>
                    <td class="tg-jkyp">Coefficient</td>
                    <td class="tg-jkyp">const</td>
                    <td class="tg-jkyp">153.737</td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-za14"></td>
                    <td class="tg-za14"></td>
                </tr>
                <tr>
                    <td class="tg-jkyp">Ridge Regression</td>
                    <td class="tg-jkyp">Coefficient</td>
                    <td class="tg-jkyp">age</td>
                    <td class="tg-jkyp">1.807</td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-za14"></td>
                    <td class="tg-za14"></td>
                </tr>
                <tr>
                    <td class="tg-jkyp">Ridge Regression</td>
                    <td class="tg-jkyp">Coefficient</td>
                    <td class="tg-jkyp">sex</td>
                    <td class="tg-jkyp">-11.448</td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-za14"></td>
                    <td class="tg-za14"></td>
                </tr>
                <tr>
                    <td class="tg-jkyp">Ridge Regression</td>
                    <td class="tg-jkyp">Coefficient</td>
                    <td class="tg-jkyp">bmi</td>
                    <td class="tg-jkyp">25.733</td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-za14"></td>
                    <td class="tg-za14"></td>
                </tr>
                <tr>
                    <td class="tg-jkyp">Ridge Regression</td>
                    <td class="tg-jkyp">Coefficient</td>
                    <td class="tg-jkyp">bp</td>
                    <td class="tg-jkyp">16.734</td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-za14"></td>
                    <td class="tg-za14"></td>
                </tr>
                <tr>
                    <td class="tg-jkyp">Ridge Regression</td>
                    <td class="tg-jkyp">Coefficient</td>
                    <td class="tg-jkyp">s1</td>
                    <td class="tg-jkyp">-34.672</td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-za14"></td>
                    <td class="tg-za14"></td>
                </tr>
                <tr>
                    <td class="tg-jkyp">Ridge Regression</td>
                    <td class="tg-jkyp">Coefficient</td>
                    <td class="tg-jkyp">s2</td>
                    <td class="tg-jkyp">17.053</td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-za14"></td>
                    <td class="tg-za14"></td>
                </tr>
                <tr>
                    <td class="tg-jkyp">Ridge Regression</td>
                    <td class="tg-jkyp">Coefficient</td>
                    <td class="tg-jkyp">s3</td>
                    <td class="tg-jkyp">3.37</td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-za14"></td>
                    <td class="tg-za14"></td>
                </tr>
                <tr>
                    <td class="tg-jkyp">Ridge Regression</td>
                    <td class="tg-jkyp">Coefficient</td>
                    <td class="tg-jkyp">s4</td>
                    <td class="tg-jkyp">11.764</td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-za14"></td>
                    <td class="tg-za14"></td>
                </tr>
                <tr>
                    <td class="tg-jkyp">Ridge Regression</td>
                    <td class="tg-jkyp">Coefficient</td>
                    <td class="tg-jkyp">s5</td>
                    <td class="tg-jkyp">31.378</td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-za14"></td>
                    <td class="tg-za14"></td>
                </tr>
                <tr>
                    <td class="tg-jkyp">Ridge Regression</td>
                    <td class="tg-jkyp">Coefficient</td>
                    <td class="tg-jkyp">s6</td>
                    <td class="tg-jkyp">2.458</td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-za14"></td>
                    <td class="tg-za14"></td>
                </tr>
            </tbody>
        </table>
    </div>

.. raw:: html

    <div style="height: 40px;"></div>


Regression Example 2
~~~~~~~~~~~~~~~~~~~~~~

In this Regression Example 2, we extend the analysis by introducing a Random Forest 
Regressor alongside Linear Regression and Ridge Regression to demonstrate how a 
model with feature importances, rather than coefficients, impacts evaluation outcomes. 
The code uses the ``summarize_model_performance`` function from ``model_metrics`` to 
assess all three models on the diabetes dataset’s test set, ensuring the Random Forest’s 
feature importance-based predictions are reflected in the results while preserving 
the coefficient-based results of the other models, as shown in the subsequent table.

.. code-block:: python

    from model_metrics import summarize_model_performance

    regression_metrics = summarize_model_performance(
        model=[linear_model, ridge_model, rf_model],
        model_title=["Linear Regression", "Ridge Regression", "Random Forest"],
        X=X_test,
        y=y_test,
        model_type="regression",
        return_df=True,
    )

    regression_metrics


**Output**

.. raw:: html

    <style type="text/css">
    .tg {
        border-collapse: collapse;
        border-spacing: 0;
        max-width: 450px; /* Fixed maximum width */
        width: 100%;
    }
    .tg td {
        border-color: black;
        border-style: solid;
        border-width: 1px;
        font-family: Arial, sans-serif;
        font-size: 12px; /* Reduced from 14px */
        overflow: hidden;
        padding: 0px 3px; /* Reduced from 0px 5px */
        word-break: normal;
    }
    .tg th {
        border-color: black;
        border-style: solid;
        border-width: 1px;
        font-family: Arial, sans-serif;
        font-size: 12px; /* Reduced from 14px */
        font-weight: normal;
        overflow: hidden;
        padding: 0px 3px; /* Reduced from 0px 5px */
        word-break: normal;
    }
    .tg .tg-oxe6 { font-weight: bold; text-align: right; vertical-align: bottom; }
    .tg .tg-jkyp { text-align: right; vertical-align: bottom; }
    .tg .tg-2b7s { text-align: right; vertical-align: bottom; }

    @media screen and (max-width: 767px) {
        .tg { width: auto !important; }
        .tg col { width: auto !important; }
        .tg-wrap { overflow-x: auto; -webkit-overflow-scrolling: touch; }
    }
    </style>

    <div class="tg-wrap">
        <table class="tg" style="table-layout: fixed; width: 450px;">
            <colgroup>
                <col style="width: 106px"> <!-- Model -->
                <col style="width: 90px"> <!-- Metric -->
                <col style="width: 60px"> <!-- Variable -->
                <col style="width: 70px"> <!-- Coefficient -->
                <col style="width: 65px"> <!-- Feat. Imp. -->
                <col style="width: 50px"> <!-- MAE -->
                <col style="width: 50px"> <!-- MAPE -->
                <col style="width: 60px"> <!-- MSE -->
                <col style="width: 50px"> <!-- RMSE -->
                <col style="width: 46px"> <!-- Expl. Var. -->
                <col style="width: 46px"> <!-- R^2 Score -->
            </colgroup>
            <thead>
                <tr>
                    <th class="tg-oxe6">Model</th>
                    <th class="tg-oxe6">Metric</th>
                    <th class="tg-oxe6">Variable</th>
                    <th class="tg-oxe6">Coefficient</th>
                    <th class="tg-oxe6">Feat. Imp.</th>
                    <th class="tg-oxe6">MAE</th>
                    <th class="tg-oxe6">MAPE</th>
                    <th class="tg-oxe6">MSE</th>
                    <th class="tg-oxe6">RMSE</th>
                    <th class="tg-oxe6">Expl. Var.</th>
                    <th class="tg-oxe6">R^2 Score</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td class="tg-jkyp">Linear Regression</td>
                    <td class="tg-jkyp">Overall Metrics</td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp">42.794</td>
                    <td class="tg-jkyp">37.5</td>
                    <td class="tg-jkyp">2900.194</td>
                    <td class="tg-jkyp">53.853</td>
                    <td class="tg-jkyp">0.455</td>
                    <td class="tg-jkyp">0.453</td>
                </tr>
                <tr>
                    <td class="tg-jkyp">Linear Regression</td>
                    <td class="tg-jkyp">Coefficient</td>
                    <td class="tg-jkyp">const</td>
                    <td class="tg-jkyp">151.346</td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                </tr>
                <tr>
                    <td class="tg-jkyp">Linear Regression</td>
                    <td class="tg-jkyp">Coefficient</td>
                    <td class="tg-jkyp">age</td>
                    <td class="tg-jkyp">37.904</td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                </tr>
                <tr>
                    <td class="tg-jkyp">Linear Regression</td>
                    <td class="tg-jkyp">Coefficient</td>
                    <td class="tg-jkyp">sex</td>
                    <td class="tg-jkyp">-241.964</td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                </tr>
                <tr>
                    <td class="tg-jkyp">Linear Regression</td>
                    <td class="tg-jkyp">Coefficient</td>
                    <td class="tg-jkyp">bmi</td>
                    <td class="tg-jkyp">542.429</td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                </tr>
                <tr>
                    <td class="tg-jkyp">Linear Regression</td>
                    <td class="tg-jkyp">Coefficient</td>
                    <td class="tg-jkyp">bp</td>
                    <td class="tg-jkyp">347.704</td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                </tr>
                <tr>
                    <td class="tg-jkyp">Linear Regression</td>
                    <td class="tg-jkyp">Coefficient</td>
                    <td class="tg-jkyp">s1</td>
                    <td class="tg-jkyp">-931.489</td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                </tr>
                <tr>
                    <td class="tg-jkyp">Linear Regression</td>
                    <td class="tg-jkyp">Coefficient</td>
                    <td class="tg-jkyp">s2</td>
                    <td class="tg-jkyp">518.062</td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                </tr>
                <tr>
                    <td class="tg-jkyp">Linear Regression</td>
                    <td class="tg-jkyp">Coefficient</td>
                    <td class="tg-jkyp">s3</td>
                    <td class="tg-jkyp">163.42</td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                </tr>
                <tr>
                    <td class="tg-jkyp">Linear Regression</td>
                    <td class="tg-jkyp">Coefficient</td>
                    <td class="tg-jkyp">s4</td>
                    <td class="tg-jkyp">275.318</td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                </tr>
                <tr>
                    <td class="tg-jkyp">Linear Regression</td>
                    <td class="tg-jkyp">Coefficient</td>
                    <td class="tg-jkyp">s5</td>
                    <td class="tg-jkyp">736.199</td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                </tr>
                <tr>
                    <td class="tg-jkyp">Linear Regression</td>
                    <td class="tg-jkyp">Coefficient</td>
                    <td class="tg-jkyp">s6</td>
                    <td class="tg-jkyp">48.671</td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                </tr>
                <tr>
                    <td class="tg-jkyp">Ridge Regression</td>
                    <td class="tg-jkyp">Overall Metrics</td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp">42.812</td>
                    <td class="tg-jkyp">37.448</td>
                    <td class="tg-jkyp">2892.015</td>
                    <td class="tg-jkyp">53.777</td>
                    <td class="tg-jkyp">0.457</td>
                    <td class="tg-jkyp">0.454</td>
                </tr>
                <tr>
                    <td class="tg-jkyp">Ridge Regression</td>
                    <td class="tg-jkyp">Coefficient</td>
                    <td class="tg-jkyp">const</td>
                    <td class="tg-jkyp">153.737</td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                </tr>
                <tr>
                    <td class="tg-jkyp">Ridge Regression</td>
                    <td class="tg-jkyp">Coefficient</td>
                    <td class="tg-jkyp">age</td>
                    <td class="tg-jkyp">1.807</td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                </tr>
                <tr>
                    <td class="tg-jkyp">Ridge Regression</td>
                    <td class="tg-jkyp">Coefficient</td>
                    <td class="tg-jkyp">sex</td>
                    <td class="tg-jkyp">-11.448</td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                </tr>
                <tr>
                    <td class="tg-jkyp">Ridge Regression</td>
                    <td class="tg-jkyp">Coefficient</td>
                    <td class="tg-jkyp">bmi</td>
                    <td class="tg-jkyp">25.733</td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                </tr>
                <tr>
                    <td class="tg-jkyp">Ridge Regression</td>
                    <td class="tg-jkyp">Coefficient</td>
                    <td class="tg-jkyp">bp</td>
                    <td class="tg-jkyp">16.734</td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                </tr>
                <tr>
                    <td class="tg-jkyp">Ridge Regression</td>
                    <td class="tg-jkyp">Coefficient</td>
                    <td class="tg-jkyp">s1</td>
                    <td class="tg-jkyp">-34.672</td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                </tr>
                <tr>
                    <td class="tg-jkyp">Ridge Regression</td>
                    <td class="tg-jkyp">Coefficient</td>
                    <td class="tg-jkyp">s2</td>
                    <td class="tg-jkyp">17.053</td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                </tr>
                <tr>
                    <td class="tg-jkyp">Ridge Regression</td>
                    <td class="tg-jkyp">Coefficient</td>
                    <td class="tg-jkyp">s3</td>
                    <td class="tg-jkyp">3.37</td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                </tr>
                <tr>
                    <td class="tg-jkyp">Ridge Regression</td>
                    <td class="tg-jkyp">Coefficient</td>
                    <td class="tg-jkyp">s4</td>
                    <td class="tg-jkyp">11.764</td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                </tr>
                <tr>
                    <td class="tg-jkyp">Ridge Regression</td>
                    <td class="tg-jkyp">Coefficient</td>
                    <td class="tg-jkyp">s5</td>
                    <td class="tg-jkyp">31.378</td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                </tr>
                <tr>
                    <td class="tg-jkyp">Ridge Regression</td>
                    <td class="tg-jkyp">Coefficient</td>
                    <td class="tg-jkyp">s6</td>
                    <td class="tg-jkyp">2.458</td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                    <td class="tg-jkyp"></td>
                </tr>
                <tr>
                    <td class="tg-2b7s">Random Forest</td>
                    <td class="tg-2b7s">Overall Metrics</td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s">44.053</td>
                    <td class="tg-2b7s">40.005</td>
                    <td class="tg-2b7s">2952.011</td>
                    <td class="tg-2b7s">54.332</td>
                    <td class="tg-2b7s">0.443</td>
                    <td class="tg-2b7s">0.443</td>
                </tr>
                <tr>
                    <td class="tg-2b7s">Random Forest</td>
                    <td class="tg-2b7s">Feat. Imp.</td>
                    <td class="tg-2b7s">age</td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s">0.059</td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                </tr>
                <tr>
                    <td class="tg-2b7s">Random Forest</td>
                    <td class="tg-2b7s">Feat. Imp.</td>
                    <td class="tg-2b7s">sex</td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s">0.01</td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                </tr>
                <tr>
                    <td class="tg-2b7s">Random Forest</td>
                    <td class="tg-2b7s">Feat. Imp.</td>
                    <td class="tg-2b7s">bmi</td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s">0.355</td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                </tr>
                <tr>
                    <td class="tg-2b7s">Random Forest</td>
                    <td class="tg-2b7s">Feat. Imp.</td>
                    <td class="tg-2b7s">bp</td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s">0.088</td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                </tr>
                <tr>
                    <td class="tg-2b7s">Random Forest</td>
                    <td class="tg-2b7s">Feat. Imp.</td>
                    <td class="tg-2b7s">s1</td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s">0.053</td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                </tr>
                <tr>
                    <td class="tg-2b7s">Random Forest</td>
                    <td class="tg-2b7s">Feat. Imp.</td>
                    <td class="tg-2b7s">s2</td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s">0.057</td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                </tr>
                <tr>
                    <td class="tg-2b7s">Random Forest</td>
                    <td class="tg-2b7s">Feat. Imp.</td>
                    <td class="tg-2b7s">s3</td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s">0.051</td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                </tr>
                <tr>
                    <td class="tg-2b7s">Random Forest</td>
                    <td class="tg-2b7s">Feat. Imp.</td>
                    <td class="tg-2b7s">s4</td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s">0.024</td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                </tr>
                <tr>
                    <td class="tg-2b7s">Random Forest</td>
                    <td class="tg-2b7s">Feat. Imp.</td>
                    <td class="tg-2b7s">s5</td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s">0.231</td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                </tr>
                <tr>
                    <td class="tg-2b7s">Random Forest</td>
                    <td class="tg-2b7s">Feat. Imp.</td>
                    <td class="tg-2b7s">s6</td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s">0.071</td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                </tr>
            </tbody>
        </table>
    </div>

.. raw:: html

    <div style="height: 40px;"></div>

Regression Example 3
~~~~~~~~~~~~~~~~~~~~~~~~~~


In some scenarios, you may want to simplify the output by excluding variables, 
coefficients, and feature importances from the model results. This example 
demonstrates how to achieve that by setting ``overall_only=True`` in the 
``summarize_model_performance`` function, producing a concise table that 
focuses on key metrics: model name, Mean Absolute Error (MAE), 
Mean Absolute Percentage Error (MAPE), Mean Squared Error (MSE), 
Root Mean Squared Error (RMSE), Explained Variance, and R² Score.

.. code-block:: python

    from model_metrics import summarize_model_performance

    regression_metrics = summarize_model_performance(
        model=[linear_model, ridge_model, rf_model],
        model_title=["Linear Regression", "Ridge Regression", "Random Forest"],
        X=X_test,
        y=y_test,
        model_type="regression",
        overall_only=True,
        return_df=True,
    )

    regression_metrics


**Output**

.. raw:: html

    <style type="text/css">
    .tg {
        border-collapse: collapse;
        border-spacing: 0;
        max-width: 450px; /* Fixed maximum width */
        width: 100%;
    }
    .tg td {
        border-color: black;
        border-style: solid;
        border-width: 1px;
        font-family: Arial, sans-serif;
        font-size: 12px; /* Reduced from 14px */
        overflow: hidden;
        padding: 0px 3px; /* Reduced from 0px 5px */
        word-break: normal;
    }
    .tg th {
        border-color: black;
        border-style: solid;
        border-width: 1px;
        font-family: Arial, sans-serif;
        font-size: 12px; /* Reduced from 14px */
        font-weight: normal;
        overflow: hidden;
        padding: 0px 3px; /* Reduced from 0px 5px */
        word-break: normal;
    }
    .tg .tg-kex3 { font-weight: bold; text-align: right; vertical-align: bottom; }
    .tg .tg-2b7s { text-align: right; vertical-align: bottom; }

    @media screen and (max-width: 767px) {
        .tg { width: auto !important; }
        .tg col { width: auto !important; }
        .tg-wrap { overflow-x: auto; -webkit-overflow-scrolling: touch; }
    }
    </style>

    <div class="tg-wrap">
        <table class="tg" style="table-layout: fixed; width: 450px;">
            <colgroup>
                <col style="width: 120px"> <!-- Model -->
                <col style="width: 120px"> <!-- Metric -->
                <col style="width: 75px"> <!-- MAE -->
                <col style="width: 75px"> <!-- MAPE -->
                <col style="width: 75px"> <!-- MSE -->
                <col style="width: 75px"> <!-- RMSE -->
                <col style="width: 75px"> <!-- Expl. Var. -->
                <col style="width: 75px"> <!-- R² Score -->
            </colgroup>
            <thead>
                <tr>
                    <th class="tg-kex3">Model</th>
                    <th class="tg-kex3">Metric</th>
                    <th class="tg-kex3">MAE</th>
                    <th class="tg-kex3">MAPE</th>
                    <th class="tg-kex3">MSE</th>
                    <th class="tg-kex3">RMSE</th>
                    <th class="tg-kex3">Expl. Var.</th>
                    <th class="tg-kex3">R^2 Score</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td class="tg-2b7s">Linear Regression</td>
                    <td class="tg-2b7s">Overall Metrics</td>
                    <td class="tg-2b7s">42.794</td>
                    <td class="tg-2b7s">37.5</td>
                    <td class="tg-2b7s">2900.194</td>
                    <td class="tg-2b7s">53.853</td>
                    <td class="tg-2b7s">0.455</td>
                    <td class="tg-2b7s">0.453</td>
                </tr>
                <tr>
                    <td class="tg-2b7s">Ridge Regression</td>
                    <td class="tg-2b7s">Overall Metrics</td>
                    <td class="tg-2b7s">42.812</td>
                    <td class="tg-2b7s">37.448</td>
                    <td class="tg-2b7s">2892.015</td>
                    <td class="tg-2b7s">53.777</td>
                    <td class="tg-2b7s">0.457</td>
                    <td class="tg-2b7s">0.454</td>
                </tr>
                <tr>
                    <td class="tg-2b7s">Random Forest</td>
                    <td class="tg-2b7s">Overall Metrics</td>
                    <td class="tg-2b7s">44.053</td>
                    <td class="tg-2b7s">40.005</td>
                    <td class="tg-2b7s">2952.011</td>
                    <td class="tg-2b7s">54.332</td>
                    <td class="tg-2b7s">0.443</td>
                    <td class="tg-2b7s">0.443</td>
                </tr>
            </tbody>
        </table>
    </div>

.. raw:: html

    <div style="height: 40px;"></div>


Lift Charts
--------------------
This section illustrates how to assess and compare the ranking effectiveness of
classification models using Lift Charts, a valuable tool for evaluating how well 
a model prioritizes positive instances relative to random chance. Leveraging the 
Logistic Regression, Decision Tree, and Random Forest Classifier models trained 
on the :ref:`synthetic dataset introduced in the Binary Classification Models section
<Binary_Classification>`, we plot Lift curves to visualize their relative ability to 
surface high-value (positive) cases at the top of the prediction list.

A Lift Chart plots the ratio of actual positives identified by the model compared 
to what would be expected by random selection, across increasingly larger proportions 
of the sample sorted by predicted probability. The baseline (Lift = 1) represents 
random chance; curves that rise above this line demonstrate the model's ability to
"lift" positive outcomes toward the top ranks. This makes Lift Charts especially 
useful in applications like marketing, fraud detection, and risk stratification—where
targeting the top segment of predictions can yield outsized value.

The ``show_lift_chart`` function enables flexible creation of Lift Charts for one or more 
models. It supports single-plot overlays, grid layouts, and detailed customization of 
labels, titles, and styling. Designed for both exploratory analysis and stakeholder 
presentation, this utility helps users better understand model ranking performance 
across the population.


.. function:: show_lift_chart(model, X, y, xlabel="Percentage of Sample", ylabel="Lift", model_title=None, overlay=False, title=None, save_plot=False, image_path_png=None, image_path_svg=None, text_wrap=None, curve_kwgs=None, linestyle_kwgs=None, grid=False, n_rows=None, n_cols=2, figsize=(8, 6), label_fontsize=12, tick_fontsize=10, gridlines=True)

    :param model: A trained model or a list of models. Each must implement ``predict_proba`` to estimate class probabilities.
    :type model: object or list[object]
    :param X: Feature matrix used to generate predictions.
    :type X: pd.DataFrame or np.ndarray
    :param y: True binary labels corresponding to the input samples.
    :type y: pd.Series or np.ndarray
    :param xlabel: Label for the x-axis. Defaults to ``"Percentage of Sample"``.
    :type xlabel: str, optional
    :param ylabel: Label for the y-axis. Defaults to ``"Lift"``.
    :type ylabel: str, optional
    :param model_title: Custom display names for the models. Can be a string or list of strings.
    :type model_title: str or list[str], optional
    :param overlay: If ``True``, overlays all model lift curves into a single plot. Defaults to ``False``.
    :type overlay: bool, optional
    :param title: Title for the plot or grid. Set to ``""`` to suppress the title. Defaults to ``None``.
    :type title: str, optional
    :param save_plot: Whether to save the chart(s) to disk. Defaults to ``False``.
    :type save_plot: bool, optional
    :param image_path_png: Output path for saving PNG image(s).
    :type image_path_png: str, optional
    :param image_path_svg: Output path for saving SVG image(s).
    :type image_path_svg: str, optional
    :param text_wrap: Maximum number of characters before wrapping titles. If ``None``, no wrapping is applied.
    :type text_wrap: int, optional
    :param curve_kwgs: Dictionary or list of dictionaries for customizing the lift curve(s) (e.g., color, linewidth).
    :type curve_kwgs: dict[str, dict] or list[dict], optional
    :param linestyle_kwgs: Styling for the baseline (random lift) reference line. Defaults to ``{"color": "gray", "linestyle": "--", "linewidth": 2}``.
    :type linestyle_kwgs: dict, optional
    :param grid: Whether to show each model in a subplot grid. Cannot be combined with ``overlay=True``.
    :type grid: bool, optional
    :param n_rows: Number of rows in the grid layout. If ``None``, automatically inferred.
    :type n_rows: int, optional
    :param n_cols: Number of columns in the grid layout. Defaults to ``2``.
    :type n_cols: int, optional
    :param figsize: Tuple specifying the size of the figure in inches. Defaults to ``(8, 6)``.
    :type figsize: tuple[int, int], optional
    :param label_fontsize: Font size for x/y-axis labels and titles. Defaults to ``12``.
    :type label_fontsize: int, optional
    :param tick_fontsize: Font size for tick marks and legend text. Defaults to ``10``.
    :type tick_fontsize: int, optional
    :param gridlines: Whether to display gridlines in plots. Defaults to ``True``.
    :type gridlines: bool, optional

    :returns: ``None.`` Displays or saves lift charts for the specified classification models.
    :rtype: ``None``

    :raises ValueError:
        - If ``overlay=True`` and ``grid=True`` are both set.


.. admonition:: Notes

    - **What is a Lift Chart?**
        - Lift quantifies how much better a model is at identifying positive cases compared to random selection.
        - The x-axis represents the proportion of the population (from highest to lowest predicted probability).
        - The y-axis shows the cumulative lift, calculated as the ratio of observed positives to expected positives under random selection.

    - **Interpreting Lift Curves:**
        - A higher and steeper curve indicates a stronger model.
        - The horizontal dashed line at ``y = 1`` is the baseline for random performance.
        - Curves that drop sharply or flatten may indicate poor ranking ability.

    - **Layout Options:**
        - Use ``overlay=True`` to visualize all models on a single axis.
        - Use ``grid=True`` for a side-by-side layout of lift charts.
        - Neither set? Each model gets its own full-sized chart.

    - **Customization:**
        - Customize the appearance of each model’s curve using ``curve_kwgs``.
        - Modify the baseline reference line with ``linestyle_kwgs``.
        - Control title wrapping and font sizes via ``text_wrap``, ``label_fontsize``, and ``tick_fontsize``.

    - **Saving Plots:**
        - If ``save_plot=True``, figures are saved as ``<model_title>_Lift.png/svg`` or ``Overlay_Lift.png/svg``.


Lift Chart Example 1 (Grid Layout)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In this first Lift Chart example, we evaluate and compare the ranking performance 
of two classification models—Logistic Regression and Random Forest Classifier—trained 
on the :ref:`synthetic dataset from the Binary Classification Models section 
<Binary_Classification>`. The chart displays Lift curves for both models in a 
two-column grid layout (``n_cols=2, n_rows=1``), enabling side-by-side comparison 
of how effectively each model prioritizes positive cases.

Each plot shows the model's Lift across increasing portions of the test set, with 
a grey dashed line at Lift = 1 indicating the baseline (random performance). Curves 
above this line reflect the model's ability to identify more positives than would 
be expected by chance. The Random Forest typically produces a steeper initial lift, 
demonstrating greater concentration of positive cases near the top-ranked predictions.

The show_lift_chart function allows for rich customization, including plot dimensions, 
axis font sizes, and curve styling. In this example, we set the line widths for both 
models and saved the plots in both PNG and SVG formats for further reporting or
documentation.

.. code-block:: python

    from model_metrics import show_lift_chart
    
    show_lift_chart(
        model=[model1, model2],
        X=X_test,
        y=y_test,
        model_title=["Logistic Regression", "Random Forest"],
        linestyle_kwgs={"color": "grey", "linestyle": "--"},
        curve_kwgs={title: {"linewidth": 2} for title in model_titles},
        grid=True,
    )


ROC AUC Curves
--------------------

This section demonstrates how to evaluate the performance of binary classification 
models using ROC AUC curves, a key metric for assessing the trade-off between 
true positive and false positive rates. Using the Logistic Regression and 
Random Forest Classifier models trained on the :ref:`synthetic dataset from the 
previous (Binary Classification Models) section <Binary_Classification>`, 
we generate ROC curves to visualize their discriminatory power.

ROC AUC (Receiver Operating Characteristic Area Under the Curve) provides a 
single scalar value representing a model's ability to distinguish between 
positive and negative classes, with a value of 1 indicating perfect classification 
and 0.5 representing random guessing. The curves are plotted by varying the 
classification threshold and calculating the true positive rate (sensitivity) 
against the false positive rate (1-specificity). This makes ROC AUC particularly 
useful for comparing models like Logistic Regression, which relies on linear 
decision boundaries, and Random Forest Classifier, which leverages ensemble 
decision trees, especially when class imbalances or threshold sensitivity are 
concerns. The ``show_roc_curve`` function simplifies this process, enabling 
users to visualize and compare these curves effectively, setting the stage for 
detailed performance analysis in subsequent examples.

The ``show_roc_curve`` function provides a flexible and powerful way to visualize 
the performance of binary classification models using Receiver Operating Characteristic 
(ROC) curves. Whether you're comparing multiple models, evaluating subgroup fairness, 
or preparing publication-ready plots, this function allows full control over layout,
styling, and annotations. It supports single and multiple model inputs, optional overlay 
or grid layouts, and group-wise comparisons via a categorical feature. Additional options 
allow custom axis labels, AUC precision, curve styling, and export to PNG/SVG. 
Designed to be both user-friendly and highly configurable, ``show_roc_curve`` 
is a practical tool for model evaluation and stakeholder communication.


.. function:: show_roc_curve(model, X, y, xlabel="False Positive Rate", ylabel="True Positive Rate", model_title=None, decimal_places=2, overlay=False, title=None, save_plot=False, image_path_png=None, image_path_svg=None, text_wrap=None, curve_kwgs=None, linestyle_kwgs=None, grid=False, n_rows=None, n_cols=2, figsize=(8, 6), label_fontsize=12, tick_fontsize=10, gridlines=True, group_category=None)

    :param model: A trained model, a string placeholder, or a list containing models or strings to evaluate.
    :type model: object or str or list[object or str]
    :param X: Feature matrix used for prediction.
    :type X: pd.DataFrame or np.ndarray
    :param y: True binary labels for evaluation.
    :type y: pd.Series or np.ndarray
    :param xlabel: Label for the x-axis. Defaults to ``"False Positive Rate"``.
    :type xlabel: str, optional
    :param ylabel: Label for the y-axis. Defaults to ``"True Positive Rate"``.
    :type ylabel: str, optional
    :param model_title: Custom title(s) for the models. Can be a string or list of strings. If ``None``, defaults to ``"Model 1"``, ``"Model 2"``, etc.
    :type model_title: str or list[str], optional
    :param decimal_places: Number of decimal places for AUC values. Defaults to ``2``.
    :type decimal_places: int, optional
    :param overlay: Whether to overlay multiple models on a single plot. Defaults to ``False``.
    :type overlay: bool, optional
    :param title: Title for the plot (used in overlay mode or as global title). If ``""``, disables the title. Defaults to ``None``.
    :type title: str, optional
    :param save_plot: Whether to save the plot(s) to file. Defaults to ``False``.
    :type save_plot: bool, optional
    :param image_path_png: File path to save the plot(s) as PNG.
    :type image_path_png: str, optional
    :param image_path_svg: File path to save the plot(s) as SVG.
    :type image_path_svg: str, optional
    :param text_wrap: Maximum character width before wrapping plot titles. If ``None``, no wrapping is applied.
    :type text_wrap: int, optional
    :param curve_kwgs: Plot styling for ROC curves. Accepts a list of dictionaries or a nested dictionary keyed by model_title.
    :type curve_kwgs: list[dict] or dict[str, dict], optional
    :param linestyle_kwgs: Style for the random guess (diagonal) line. Defaults to ``{"color": "gray", "linestyle": "--", "linewidth": 2}``.
    :type linestyle_kwgs: dict, optional
    :param grid: Whether to organize the ROC plots in a subplot grid layout. Cannot be used with ``overlay=True`` or ``group_category``.
    :type grid: bool, optional
    :param n_rows: Number of rows in the grid layout. If ``None``, calculated automatically based on number of models and columns.
    :type n_rows: int, optional
    :param n_cols: Number of columns in the grid layout. Defaults to ``2``.
    :type n_cols: int, optional
    :param figsize: Size of the plot or grid of plots, in inches. Defaults to ``(8, 6)``.
    :type figsize: tuple, optional
    :param label_fontsize: Font size for axis labels and titles. Defaults to ``12``.
    :type label_fontsize: int, optional
    :param tick_fontsize: Font size for ticks and legend text. Defaults to ``10``.
    :type tick_fontsize: int, optional
    :param gridlines: Whether to display grid lines on plots. Defaults to ``True``.
    :type gridlines: bool, optional
    :param group_category: Categorical array to group ROC curves. Cannot be used with ``grid=True`` or ``overlay=True``.
    :type group_category: array-like, optional

    :returns: ``None.`` Displays or saves ROC curve plots for classification models.
    :rtype: ``None``

    :raises ValueError:
        - If ``grid=True`` and ``overlay=True`` are both set.
        - If ``group_category`` is used with ``grid`` or ``overlay``.
        - If ``overlay=True`` is used with only one model.

.. admonition:: Notes

    - **Flexible Inputs:**
        - ``model`` and ``model_title`` can be individual items or lists. Strings passed in ``model`` are treated as placeholder names.
        - Titles can be automatically inferred or explicitly passed using ``model_title``.

    - **Group-Wise ROC:**
        - If ``group_category`` is passed, separate ROC curves are plotted for each unique group.
        - The legend will include group-specific AUC and class distribution (e.g., ``AUC = 0.87, Count: 500, Pos: 120, Neg: 380``).

    - **Plot Modes:**
        - ``overlay=True`` overlays all models in one figure.
        - ``grid=True`` arranges individual ROC plots in a subplot layout.
        - If neither is set, separate full-size plots are shown for each model.

    - **Legend and Styling:**
        - A random guess reference line (diagonal) is plotted by default.
        - Customize ROC curves with ``curve_kwgs`` and the diagonal line with ``linestyle_kwgs``.
        - Titles can be disabled with ``title=""``.

    - **Saving Plots:**
        - If ``save_plot=True``, plots are saved using the base filename format ``<model_name>_roc_auc`` or ``overlay_roc_auc_plot``.

The ``show_roc_curve`` function provides flexible and highly customizable 
plotting of ROC curves for binary classification models. It supports overlays, 
grid layouts, and subgroup visualizations, while also allowing export options 
and styling hooks for publication-ready output.


ROC AUC Example 1 (Grid Layout)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In this first ROC AUC evaluation example, we plot the ROC curves for two 
models: Logistic Regression and Random Forest Classifier, trained on the 
:ref:`synthetic dataset from the Binary Classification Models section 
<Binary_Classification>`. The curves are displayed side by side 
using a grid layout (``n_cols=2, n_rows=1``), with the Logistic Regression curve 
in blue and the Random Forest curve in green for clear differentiation. 
A red dashed line represents the random guessing baseline. This example 
demonstrates how the ``show_roc_curve`` function enables straightforward 
visualization of model performance, with options to customize colors, 
add a grid, and save the plot for reporting purposes.

.. code-block:: python

    from model_metrics import show_roc_curve

    show_roc_curve(
        model=[model1, model2],
        X=X_test,
        y=y_test,
        model_title=model_title,
        decimal_places=2,
        n_cols=2,
        n_rows=1,
        curve_kwgs={
            "Logistic Regression": {"color": "blue", "linewidth": 2},
            "Random Forest": {"color": "green", "linewidth": 2},
        },
        linestyle_kwgs={"color": "red", "linestyle": "--"},
        grid=True,
        figsize=(12, 6),
    )

**Output**

.. raw:: html

   <div class="no-click">

.. image:: ../assets/grid_roc_auc_plot.svg
   :alt: ROC AUC Example 1
   :align: center
   :width: 900px

.. raw:: html

    <div style="height: 40px;"></div>

ROC AUC Example 2 (Overlay)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In this second ROC AUC evaluation example, we focus on overlaying the results of 
two models—Logistic Regression and Random Forest Classifier—trained on the 
:ref:`synthetic dataset from the Binary Classification Models section 
<Binary_Classification>` onto a single plot. Using the ``show_roc_curve`` 
function with the ``overlay=True`` parameter, the ROC curves for both models are 
displayed together, with Logistic Regression in blue and Random Forest in black, 
both with a ``linewidth=2``. A red dashed line serves as the random guessing 
baseline, and the plot includes a custom title for clarity.


.. code-block:: python

    from model_metrics import show_roc_curve

    show_roc_curve(
        model=[model1, model2],
        X=X_test,
        y=y_test,
        model_title=model_title,
        decimal_places=2,
        curve_kwgs={
            "Logistic Regression": {"color": "blue", "linewidth": 2},
            "Random Forest": {"color": "black", "linewidth": 2},
        },
        linestyle_kwgs={"color": "red", "linestyle": "--"},
        title="ROC Curves: Logistic Regression and Random Forest",
        overlay=True,
    )

**Output**

.. raw:: html

   <div class="no-click">

.. image:: ../assets/overlay_roc_auc.svg
   :alt: ROC AUC Example 2
   :align: center
   :width: 850px

.. raw:: html

    <div style="height: 40px;"></div>

ROC AUC Example 3 (by Category)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In this third ROC AUC evaluation example, we utilize the well-known 
*Adult Income* dataset [2]_, a widely used benchmark for binary classification 
tasks. Its rich mix of categorical and numerical features makes it particularly 
suitable for analyzing model performance across different subgroups.

To build and evaluate our models, we use the ``model_tuner`` library [3]_. 
:ref:`Click here to view the corresponding codebase for this workflow. <Adult_Income_Training>`

The objective here is to assess ROC AUC scores not just overall, but 
**across each category of a selected feature**—such as *occupation*, 
*education*, *marital-status*, or *race*. This approach enables deeper insight into how 
performance varies by subgroup, which is particularly important for fairness, 
bias detection, and subgroup-level interpretability.

The ``show_roc_curve`` function supports this analysis through the 
``group_category`` parameter. 

For example, by passing ``group_category=X_test_2["race"]``, 
you can generate a separate ROC curve for each unique racial group in the dataset:


.. code-block:: python

    from model_metrics import show_roc_curve
    
    show_roc_curve(
        model=model_dt["model"].estimator,
        X=X_test,
        y=y_test,
        model_title="Decision Tree Classifier,
        decimal_places=2,
        group_category=X_test_2["race"],
    )

**Output**

.. raw:: html

   <div class="no-click">


.. image:: ../assets/decision_tree_classifier_race_roc_auc.svg
    :alt: Decision Tree ROC AUC
    :width: 850px
    :align: center

.. raw:: html

    <div style="height: 40px;"></div>


Precision-Recall Curves
------------------------

This section demonstrates how to evaluate the performance of binary classification 
models using Precision-Recall (PR) curves, a critical visualization for understanding 
model behavior in the presence of class imbalance. Using the Logistic Regression 
and Random Forest Classifier models trained on the 
:ref:`synthetic dataset from the previous (Binary Classification Models) section <Binary_Classification>`, 
we generate PR curves to examine how well each model identifies true positives while limiting false positives.

Precision-Recall curves focus on the trade-off between **precision** 
(positive predictive value) and **recall** (sensitivity) across different 
classification thresholds. This is particularly important when the positive 
class is rare—as is common in fraud detection, disease diagnosis, or adverse 
event prediction—because ROC AUC can overstate performance under imbalance. 
Unlike the ROC curve, the PR curve is sensitive to the proportion of positive 
examples and gives a clearer picture of how well a model performs where it 
matters most: in identifying the positive class.

The **area under the Precision-Recall curve**, also known as Average Precision 
(AP), summarizes model performance across thresholds. A model that maintains high 
precision as recall increases is generally more desirable, especially in settings 
where false positives have a high cost. This makes the PR curve a complementary 
and sometimes more informative tool than ROC AUC in skewed classification scenarios.


.. function:: show_pr_curve(model, X, y, xlabel="Recall", ylabel="Precision", model_title=None, decimal_places=2, overlay=False, title=None, save_plot=False, image_path_png=None, image_path_svg=None, text_wrap=None, curve_kwgs=None, grid=False, n_rows=None, n_cols=2, figsize=(8, 6), label_fontsize=12, tick_fontsize=10, gridlines=True, group_category=None)

    :param model: A trained model, a string placeholder, or a list containing models or strings to evaluate.
    :type model: object or str or list[object or str]
    :param X: Feature matrix used for prediction.
    :type X: pd.DataFrame or np.ndarray
    :param y: True binary labels for evaluation.
    :type y: pd.Series or np.ndarray
    :param xlabel: Label for the x-axis. Defaults to ``"Recall"``.
    :type xlabel: str, optional
    :param ylabel: Label for the y-axis. Defaults to ``"Precision"``.
    :type ylabel: str, optional
    :param model_title: Custom title(s) for the model(s). Can be a string or list of strings. If ``None``, defaults to ``"Model 1"``, ``"Model 2"``, etc.
    :type model_title: str or list[str], optional
    :param decimal_places: Number of decimal places for Average Precision (AP) values. Defaults to ``2``.
    :type decimal_places: int, optional
    :param overlay: Whether to overlay multiple models on a single plot. Defaults to ``False``.
    :type overlay: bool, optional
    :param title: Title for the plot (used in overlay mode or as global title). If ``""``, disables the title. Defaults to ``None``.
    :type title: str, optional
    :param save_plot: Whether to save the plot(s) to file. Defaults to ``False``.
    :type save_plot: bool, optional
    :param image_path_png: File path to save the plot(s) as PNG.
    :type image_path_png: str, optional
    :param image_path_svg: File path to save the plot(s) as SVG.
    :type image_path_svg: str, optional
    :param text_wrap: Maximum character width before wrapping plot titles. If ``None``, no wrapping is applied.
    :type text_wrap: int, optional
    :param curve_kwgs: Plot styling for PR curves. Accepts a list of dictionaries or a nested dictionary keyed by model_title.
    :type curve_kwgs: list[dict] or dict[str, dict], optional
    :param grid: Whether to organize the PR plots in a subplot grid layout. Cannot be used with ``overlay=True`` or ``group_category``.
    :type grid: bool, optional
    :param n_rows: Number of rows in the grid layout. If ``None``, calculated automatically based on number of models and columns.
    :type n_rows: int, optional
    :param n_cols: Number of columns in the grid layout. Defaults to ``2``.
    :type n_cols: int, optional
    :param figsize: Size of the plot or grid of plots, in inches. Defaults to ``(8, 6)``.
    :type figsize: tuple, optional
    :param label_fontsize: Font size for axis labels and titles. Defaults to ``12``.
    :type label_fontsize: int, optional
    :param tick_fontsize: Font size for ticks and legend text. Defaults to ``10``.
    :type tick_fontsize: int, optional
    :param gridlines: Whether to display grid lines on plots. Defaults to ``True``.
    :type gridlines: bool, optional
    :param group_category: Categorical array to group PR curves. Cannot be used with ``grid=True`` or ``overlay=True``.
    :type group_category: array-like, optional
    :param legend_metric: Metric to display in the legend. Either ``"ap"`` (Average Precision) or ``"aucpr"`` (area under the PR curve). Defaults to ``"ap"``.
    :type legend_metric: str, optional

    :returns: ``None.`` Displays or saves Precision-Recall curve plots for classification models.
    :rtype: ``None``

    :raises ValueError:
        - If ``grid=True`` and ``overlay=True`` are both set.
        - If ``group_category`` is used with ``grid=True`` or ``overlay=True``.
        - If ``overlay=True`` is used with only one model.
        - If ``legend_metric`` is not one of ``"ap"`` or ``"aucpr"``.
    :raises TypeError:
        - If ``model_title`` is not a string, list of strings, or ``None``.

.. admonition:: Notes

    - **Flexible Inputs:**
        - ``model`` and ``model_title`` can be individual items or lists. Strings passed in ``model`` are treated as placeholder names.
        - Titles can be automatically inferred or explicitly passed using ``model_title``.

    - **Group-Wise PR:**
        - If ``group_category`` is passed, separate PR curves are plotted for each unique group.
        - The legend will include group-specific Average Precision and class distribution (e.g., ``AP = 0.78, Count: 500, Pos: 120, Neg: 380``).

    - **Average Precision vs. AUCPR:**
        - By default, the legend shows **Average Precision (AP)**, which summarizes the PR curve with greater emphasis on the performance at higher precision levels.
        - If the user passes ``legend_metric="aucpr"``, the legend will instead display **AUCPR** (Area Under the Precision-Recall Curve), which gives equal weight to all parts of the curve.
  
    - **Plot Modes:**
        - ``overlay=True`` overlays all models in one figure.
        - ``grid=True`` arranges individual PR plots in a subplot layout.
        - If neither is set, separate full-size plots are shown for each model.

    - **Legend and Styling:**
        - A random classifier baseline (constant precision) is plotted by default.
        - Customize PR curves with ``curve_kwgs``.
        - Titles can be disabled with ``title=""``.

    - **Saving Plots:**
        - If ``save_plot=True``, plots are saved using the base filename format ``<model_name>_precision_recall`` or ``overlay_pr_plot``.

The ``show_pr_curve`` function provides flexible and highly customizable plotting 
of Precision-Recall curves for binary classification models. It supports overlays, 
grid layouts, and subgroup visualizations, while also allowing export options and 
styling hooks for publication-ready output.


Precision-Recall Example 1 (Grid Layout)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In this first Precision-Recall evaluation example, we plot the PR curves for two 
models: Logistic Regression and Random Forest Classifier, both trained on the 
:ref:`synthetic dataset from the Binary Classification Models section <Binary_Classification>`. 
The curves are arranged side by side using a grid layout (``n_cols=2, n_rows=1``), 
with the Logistic Regression curve rendered in blue and the Random Forest curve 
in green to distinguish between models. A gray dashed line indicates the baseline 
precision, equal to the prevalence of the positive class in the dataset.

This example illustrates how the ``show_pr_curve`` function makes it easy to 
visualize and compare model performance when dealing with class imbalance. It 
also demonstrates layout flexibility and customization options, including gridlines, 
label styling, and export functionality—making it suitable for both exploratory 
analysis and final reporting.

.. code-block:: python

    from model_metrics import show_pr_curve

    show_pr_curve(
        model=[logistic_model, rf_model],
        X=X_test,
        y=y_test,
        model_title=["Logistic Regression", "Random Forest"],
        decimal_places=2,
        grid=True,
        n_cols=2,
        n_rows=1,
        curve_kwgs=[
            {"color": "blue"},
            {"color": "green"}
        ],
        gridlines=True
    )

**Output**

.. raw:: html

   <div class="no-click">


.. image:: ../assets/grid_pr_plot.svg
    :alt: Precision-Recall Curve Example 1
    :width: 900px
    :align: center

.. raw:: html

    <div style="height: 40px;"></div>

Precision-Recall Example 2 (Overlay)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In this second Precision-Recall evaluation example, we focus on overlaying the 
results of two models—Logistic Regression and Random Forest Classifier—trained 
on the :ref:`synthetic dataset from the Binary Classification Models section 
<Binary_Classification>` onto a single plot. Using the ``show_pr_curve`` 
function with the ``overlay=True`` parameter, the Precision-Recall curves for 
both models are displayed together, with Logistic Regression in blue and Random 
Forest in black, both with a ``linewidth=2``. The plot includes a custom title 
for clarity.


.. code-block:: python

    from model_metrics import show_pr_curve

    show_pr_curve(
        model=[model1, model2],
        X=X_test,
        y=y_test,
        model_title=model_title,
        curve_kwgs={
            "Logistic Regression": {"color": "blue", "linewidth": 2},
            "Random Forest": {"color": "black", "linewidth": 2},
        },
        title="ROC Curves: Logistic Regression and Random Forest",
        overlay=True,
    )

**Output**

.. raw:: html

   <div class="no-click">


.. image:: ../assets/overlay_pr_plot.svg
    :alt: Precision-Recall Curve Example 2
    :width: 900px
    :align: center

.. raw:: html

    <div style="height: 40px;"></div>



Precision-Recall Example 3 (Categorical)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In this third Precision-Recall evaluation example, we utilize the well-known 
*Adult Income* dataset [2]_, a widely used benchmark for binary classification 
tasks. Its rich mix of categorical and numerical features makes it particularly 
suitable for analyzing model performance across different subgroups.

To build and evaluate our models, we use the ``model_tuner`` library [3]_. 
:ref:`Click here to view the corresponding codebase for this workflow. <Adult_Income_Training>`

The objective here is to assess ROC AUC scores not just overall, but 
**across each category of a selected feature**—such as *occupation*, 
*education*, *marital-status*, or *race*. This approach enables deeper insight into how 
performance varies by subgroup, which is particularly important for fairness, 
bias detection, and subgroup-level interpretability.

The ``show_pr_curve`` function supports this analysis through the 
``group_category`` parameter. 

For example, by passing ``group_category=X_test_2["race"]``, 
you can generate a separate ROC curve for each unique racial group in the dataset:


.. code-block:: python

    from model_metrics import show_pr_curve
    
    show_pr_curve(
        model=model_dt["model"].estimator,
        X=X_test,
        y=y_test,
        model_title="Decision Tree Classifier,
        group_category=X_test_2["race"],
        legend_metric="aucpr",
    )

**Output**

.. raw:: html

   <div class="no-click">


.. image:: ../assets/decision_tree_classifier_precision_recall_race.svg
    :alt: Decision Tree Precision-Recall Example 3
    :width: 950px
    :align: center

.. raw:: html

    <div style="height: 40px;"></div>


.. _confusion_matrix_evaluation:

Confusion Matrix Evaluation
-----------------------------

This section introduces the ``show_confusion_matrix`` function, which provides a 
flexible, styled interface for generating and visualizing confusion matrices 
across one or more classification models. It supports advanced features like 
threshold overrides, subgroup labeling, classification report display, and fully 
customizable plot aesthetics including grid layouts.

The confusion matrix is a fundamental diagnostic tool for classification models, 
displaying the counts of true positives, true negatives, false positives, and 
false negatives. This function goes beyond standard implementations by allowing 
for custom thresholds (globally or per model), label annotation (e.g., TP, FP, etc.), 
plot exporting, colorbar toggling, and grid visualization.

This is especially useful when comparing multiple models side-by-side or needing 
publication-ready confusion matrices for stakeholders.

.. function:: show_confusion_matrix(model, X, y, model_title=None, title=None, model_threshold=None, custom_threshold=None, class_labels=None, cmap="Blues", save_plot=False, image_path_png=None, image_path_svg=None, text_wrap=None, figsize=(8, 6), labels=True, label_fontsize=12, tick_fontsize=10, inner_fontsize=10, grid=False, score=None, class_report=False, **kwargs)

    :param model: A single model (object or string), or a list of models or string placeholders.
    :type model: object or str or list[object or str]
    :param X: Feature matrix used for prediction.
    :type X: pd.DataFrame or np.ndarray
    :param y: True target labels.
    :type y: pd.Series or np.ndarray
    :param model_title: Custom title(s) for each model. Can be a string or list of strings. If None, defaults to ``"Model 1"``, ``"Model 2"``, etc.
    :type model_title: str or list[str], optional
    :param title: Title for each plot. If ``""``, no title is displayed. If None, a default title is shown.
    :type title: str, optional
    :param model_threshold: Dictionary of thresholds keyed by model title. Used if ``custom_threshold`` is not set.
    :type model_threshold: dict, optional
    :param custom_threshold: Global override threshold to apply across all models.
    :type custom_threshold: float, optional
    :param class_labels: Custom labels for the classes in the matrix.
    :type class_labels: list[str], optional
    :param cmap: Colormap to use for the heatmap. Defaults to ``"Blues"``.
    :type cmap: str, optional
    :param save_plot: Whether to save the generated plot(s).
    :type save_plot: bool, optional
    :param image_path_png: Path to save the PNG version of the image.
    :type image_path_png: str, optional
    :param image_path_svg: Path to save the SVG version of the image.
    :type image_path_svg: str, optional
    :param text_wrap: Maximum width of plot titles before wrapping.
    :type text_wrap: int, optional
    :param figsize: Figure size in inches. Defaults to ``(8, 6)``.
    :type figsize: tuple[int, int], optional
    :param labels: Whether to annotate matrix cells with ``TP``, ``FP``, ``FN``, ``TN``.
    :type labels: bool, optional
    :param label_fontsize: Font size for axis labels and titles.
    :type label_fontsize: int, optional
    :param tick_fontsize: Font size for axis ticks.
    :type tick_fontsize: int, optional
    :param inner_fontsize: Font size for numbers and labels inside cells.
    :type inner_fontsize: int, optional
    :param grid: Whether to display multiple models in a grid layout.
    :type grid: bool, optional
    :param score: Scoring metric to use when optimizing threshold (if applicable).
    :type score: str, optional
    :param class_report: If True, prints a classification report below each matrix.
    :type class_report: bool, optional
    :param kwargs: Additional keyword arguments for customization (e.g., show_colorbar, ``n_cols``).
    :type kwargs: dict, optional

    :returns: None. Displays confusion matrix plots (and optionally saves them).
    :rtype: None

    :raises TypeError: If ``model_title`` is not a string, a list of strings, or None.

.. admonition:: Notes

    - **Model Support:**
        - Supports single or multiple classification models.
        - ``model_title`` may be inferred automatically or provided explicitly.

    - **Threshold Handling:**
        - Use ``model_threshold`` to specify per-model thresholds.
        - ``custom_threshold`` overrides all other thresholds.

    - **Plotting Modes:**
        - ``grid=True`` arranges plots in subplots.
        - Otherwise, plots are displayed one at a time.

    - **Labeling:**
        - Set ``labels=False`` to disable annotating cells with ``TP``, ``FP``, ``FN``, ``TN``.
        - Always shows raw numeric values inside cells.

    - **Colorbar & Styling:**
        - Toggle colorbar via ``show_colorbar`` (passed via ``kwargs``).
        - Colormap and font sizes are fully configurable.

    - **Exporting Plots:**
        - Plots can be saved as both PNG and SVG using the respective paths.
        - Saved filenames follow the pattern ``Confusion_Matrix_<model_name>`` or ``Grid_Confusion_Matrix``.

.. _confusion_matrix_example_1:

Confusion Matrix Example 1 (Threshold=0.5)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In this first confusion matrix evaluation example, we focus on showing the 
results of two models—Logistic Regression and Random Forest Classifier—trained 
on the :ref:`synthetic dataset from the Binary Classification Models section 
<Binary_Classification>` onto a single plot.

.. code-block:: python

    from model_metrics import show_confusion_matrix

    show_confusion_matrix(
        model=[model1, model2],
        X=X_test,
        y=y_test,
        model_title=model_title,
        cmap="Blues",
        text_wrap=20,
        grid=True,
        n_cols=2,
        n_rows=1,
        figsize=(6, 6),
    )

**Output**

.. raw:: html

   <div class="no-click">


.. image:: ../assets/conf_matrix_1.svg
    :alt: Confusion Matrix Example 1
    :width: 900px
    :align: center

.. raw:: html

    <div style="height: 40px;"></div>

Confusion Matrix Example 2 (Classification Report)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This second confusion matrix evaluation example is nearly identical to the :ref:`first <confusion_matrix_example_1>`, 
but uses a different color map (``cmap="viridis"``) and sets ``class_report=True`` 
to print classification reports for each model in addition to the visual output.

.. code-block:: python

    from model_metrics import show_confusion_matrix

    show_confusion_matrix(
        model=[model1, model2],
        X=X_test,
        y=y_test,
        model_title=model_title,
        cmap="viridis",
        text_wrap=20,
        grid=True,
        n_cols=2,
        n_rows=1,
        figsize=(6, 6),
        class_report=True
    )

**Output**

.. code-block:: text

    Confusion Matrix for Logistic Regression: 

              Predicted 0  Predicted 1
    Actual 0           76           18
    Actual 1           13           93

    Classification Report for Logistic Regression: 

                  precision    recall  f1-score   support

               0       0.85      0.81      0.83        94
               1       0.84      0.88      0.86       106

        accuracy                           0.84       200
       macro avg       0.85      0.84      0.84       200
    weighted avg       0.85      0.84      0.84       200

    Confusion Matrix for Random Forest: 

               Predicted 0  Predicted 1
    Actual 0            84           10
    Actual 1             3          103

    Classification Report for Random Forest: 

                  precision    recall  f1-score   support

               0       0.97      0.89      0.93        94
               1       0.91      0.97      0.94       106

        accuracy                           0.94       200
       macro avg       0.94      0.93      0.93       200
    weighted avg       0.94      0.94      0.93       200


.. raw:: html

   <div class="no-click">


.. image:: ../assets/conf_matrix_2.svg
    :alt: Confusion Matrix Example 2
    :align: center


.. raw:: html

    <div style="height: 40px;"></div>



Confusion Matrix Example 3 (Threshold = 0.37)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In this third confusion matrix evaluation example using the :ref:`synthetic dataset 
from the Binary Classification Models section <Binary_Classification>`, we apply 
a custom classification threshold of 0.37 using the ``custom_threshold`` parameter. 
This overrides the default threshold of 0.5 and enables us to inspect how the 
confusion matrices shift when a more lenient decision boundary is applied. Refer 
to the section on :ref:`threshold selection logic <threshold_selection_logic>` 
for caveats on choosing the right threshold.

This is especially useful in imbalanced classification problems or cost-sensitive 
environments where the trade-off between precision and recall must be adjusted. 
By lowering the threshold, we typically increase the number of positive predictions, 
which can improve recall but may come at the cost of more false positives.

The output matrices for both models—Logistic Regression and Random Forest—are shown 
side by side in a grid layout for easy visual comparison.

.. code-block:: python

    from model_metrics import show_confusion_matrix

    show_confusion_matrix(
        model=[model1, model2],
        X=X_test,
        y=y_test,
        model_title=model_title,
        text_wrap=20,
        grid=True,
        n_cols=2,
        n_rows=1,
        figsize=(6, 6),
        custom_threshold=0.37,
    )


**Output**

.. raw:: html

   <div class="no-click">


.. image:: ../assets/conf_matrix_3.svg
    :alt: Confusion Matrix Example 3
    :width: 900px
    :align: center


.. raw:: html

    <div style="height: 40px;"></div>


Calibration Curves
----------------------

This section focuses on calibration curves, a diagnostic tool that compares 
predicted probabilities to actual outcomes, helping evaluate how well a model's 
predicted confidence aligns with observed frequencies. Using models like Logistic 
Regression or Random Forest on the :ref:`synthetic dataset from the previous 
(Binary Classification Models) section <Binary_Classification>`, we generate 
calibration curves to assess the reliability of model probabilities.

Calibration is especially important in domains where probability outputs inform 
downstream decisions, such as healthcare, finance, and risk management. A 
well-calibrated model not only predicts the correct class but also outputs 
meaningful probabilities—for example, when a model predicts a 0.7 probability, 
we expect roughly 70% of such predictions to be correct.

The ``show_calibration_curve`` function simplifies this process by allowing users to 
visualize calibration performance across models or subgroups. The plots show the 
mean predicted probabilities against the actual observed fractions of positive 
cases, with an optional reference line representing perfect calibration. 
Additional features include support for overlay or grid layouts, subgroup 
analysis by categorical features, and optional Brier score display—a scalar 
measure of calibration quality.

The function offers full control over styling, figure layout, axis labels, and 
output format, making it easy to generate both exploratory and publication-ready 
plots.

.. function:: show_calibration_curve(model, X, y, xlabel="Mean Predicted Probability", ylabel="Fraction of Positives", model_title=None, overlay=False, title=None, save_plot=False, image_path_png=None, image_path_svg=None, text_wrap=None, curve_kwgs=None, grid=False, n_cols=2, n_rows=None, figsize=None, label_fontsize=12, tick_fontsize=10, bins=10, marker="o", show_brier_score=True, gridlines=True, linestyle_kwgs=None, group_category=None, **kwargs)

    :param model: A trained classifier or a list of classifiers to evaluate.
    :type model: estimator or list
    :param X: Feature matrix used for predictions.
    :type X: pd.DataFrame or np.ndarray
    :param y: True binary target values.
    :type y: pd.Series or np.ndarray
    :param xlabel: X-axis label. Defaults to ``"Mean Predicted Probability"``.
    :type xlabel: str, optional
    :param ylabel: Y-axis label. Defaults to ``"Fraction of Positives"``.
    :type ylabel: str, optional
    :param model_title: Custom title(s) for the models.
    :type model_title: str or list[str], optional
    :param overlay: If ``True``, overlays multiple models on one plot.
    :type overlay: bool, optional
    :param title: Title for the plot. Use ``""`` to suppress.
    :type title: str, optional
    :param save_plot: Whether to save the plot(s).
    :type save_plot: bool, optional
    :param image_path_png: Directory path for PNG export.
    :type image_path_png: str, optional
    :param image_path_svg: Directory path for SVG export.
    :type image_path_svg: str, optional
    :param text_wrap: Max characters before title text wraps.
    :type text_wrap: int, optional
    :param curve_kwgs: Styling options for the calibration curves.
    :type curve_kwgs: list[dict] or dict[str, dict], optional
    :param grid: Whether to arrange models in a subplot grid.
    :type grid: bool, optional
    :param n_cols: Number of columns in the grid layout. Defaults to ``2``.
    :type n_cols: int, optional
    :param n_rows: Number of rows in the grid layout. Auto-calculated if ``None``.
    :type n_rows: int, optional
    :param figsize: Figure size in inches (width, height).
    :type figsize: tuple, optional
    :param label_fontsize: Font size for axis labels and titles.
    :type label_fontsize: int, optional
    :param tick_fontsize: Font size for ticks and legend entries.
    :type tick_fontsize: int, optional
    :param bins: Number of bins used to compute calibration.
    :type bins: int, optional
    :param marker: Marker style for calibration points.
    :type marker: str, optional
    :param show_brier_score: Whether to display Brier score in the legend.
    :type show_brier_score: bool, optional
    :param gridlines: Whether to show gridlines on plots.
    :type gridlines: bool, optional
    :param linestyle_kwgs: Styling for the "perfectly calibrated" reference line.
    :type linestyle_kwgs: dict, optional
    :param group_category: Categorical variable used to create subgroup calibration plots.
    :type group_category: array-like, optional

    :returns: ``None.`` Displays or saves calibration plots for classification models.
    :rtype: ``None``

    :raises ValueError:
        - If ``overlay=True`` and ``grid=True`` are both set.
        - If ``group_category`` is used with ``overlay`` or ``grid``.
        - If ``curve_kwgs`` list does not match number of models.

.. admonition:: Notes

    - **Calibration vs Discrimination:**
        - Calibration evaluates how well predicted probabilities reflect observed outcomes, while ROC AUC measures a model's ability to rank predictions.

    - **Flexible Plotting Modes:**
        - ``overlay=True`` plots multiple models on one figure.
        - ``grid=True`` arranges plots in a grid layout.
        - If neither is set, individual full-size plots are created.

    - **Group-Wise Analysis:**
        - Passing ``group_category`` plots separate calibration curves by subgroup (e.g., age, race).
        - Each subgroup's Brier score is shown when ``show_brier_score=True``.

    - **Customization:**
        - Use ``curve_kwgs`` and ``linestyle_kwgs`` to control styling.
        - Add markers, gridlines, and custom titles to suit report or presentation needs.

    - **Saving Outputs:**
        - Set ``save_plot=True`` and specify ``image_path_png`` or ``image_path_svg`` to export figures.
        - Filenames are auto-generated based on model name and plot type.

.. important::

    Calibration curves are a valuable diagnostic tool for assessing the alignment 
    between predicted probabilities and actual outcomes. By plotting the fraction 
    of positives against predicted probabilities, we can evaluate how well a model's 
    confidence scores correspond to observed reality. While these plots offer important 
    insights, it's equally important to understand the :ref:`assumptions and limitations behind 
    the calibration methods used <caveats_in_calibration>`.


Calibration Curve Example 1 (Grid-like)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This example presents calibration curves for two classification models trained on  
the well-known *Adult Income* dataset [2]_, a widely used benchmark for binary 
classification tasks. Its rich mix of categorical and numerical features makes 
it particularly suitable for analyzing model performance across different subgroups.

To build and evaluate our models, we use the ``model_tuner`` library [3]_. 
:ref:`Click here to view the corresponding codebase for this workflow. <Adult_Income_Training>` 
The classification models are displayed side by side in a grid layout. Each 
subplot shows how well the predicted probabilities from a model align with the 
actual observed outcomes. A diagonal dashed line representing perfect calibration 
is included in both plots, and Brier scores are shown in the legend to quantify 
each model's calibration accuracy.

By setting ``grid=True``, the function automatically arranges the individual plots 
based on the number of models and specified columns. This layout is ideal for 
visually comparing calibration behavior across models without overlapping lines.

.. code-block:: python

    from model_metrics import show_calibration_curve

    show_calibration_curve(
        model=pipelines_or_models[:2],
        X=X_test,
        y=y_test,
        model_title=model_titles[:2],
        text_wrap=50,
        figsize=(12, 6),
        label_fontsize=16,
        tick_fontsize=13,
        bins=10,
        show_brier_score=True,
        grid=True,
        linestyle_kwgs={"color": "black"},
    )


**Output**

.. raw:: html

   <div class="no-click">


.. image:: ../assets/calibration_1.svg
    :alt: Calibration Curves Example 1
    :width: 900px
    :align: center


.. raw:: html

    <div style="height: 40px;"></div>


Calibration Curve Example 2 (Overlay)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This example also uses the well-known *Adult Income* dataset [2]_, a widely used 
benchmark for binary classification tasks. Its rich mix of categorical and 
numerical features makes it particularly suitable for analyzing model performance 
across different subgroups.

To build and evaluate our models, we use the ``model_tuner`` library [3]_. 
:ref:`Click here to view the corresponding codebase for this workflow. <Adult_Income_Training>`
This example demonstrates how to overlay calibration curves from multiple classification 
models in a single plot. Overlaying allows for direct visual comparison of how predicted 
probabilities from each model align with actual outcomes on the same axes.

The diagonal dashed line represents perfect calibration, and Brier scores are included 
in the legend for each model, providing a quantitative measure of calibration accuracy.

By setting ``overlay=True``, the function combines all model curves into one figure, 
making it easier to evaluate relative performance without splitting across subplots.

.. code-block:: python

    from model_metrics import show_calibration_curve

    show_calibration_curve(
        model=pipelines_or_models[:2],
        X=X_test,
        y=y_test,
        model_title=model_titles[:2],
        bins=10,
        show_brier_score=True,
        overlay=True
        linestyle_kwgs={"color": "black"},
    )


**Output**

.. raw:: html

   <div class="no-click">


.. image:: ../assets/calibration_2.svg
    :alt: Calibration Curves Example 2
    :width: 900px
    :align: center


.. raw:: html

    <div style="height: 40px;"></div>


Calibration Curve Example 3 (by Category) 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This example, too, uses the well-known *Adult Income* dataset [2]_, a widely used 
benchmark for binary classification tasks. Its rich mix of categorical and numerical 
features makes it particularly suitable for analyzing model performance across 
different subgroups.

To build and evaluate our models, we use the ``model_tuner`` library [3]_. 
:ref:`Click here to view the corresponding codebase for this workflow. <Adult_Income_Training>`
This example shows how to visualize calibration curves separately for each
category within a given feature—in this case, the race column of the joined
test set—using a single Random Forest classifier. Each plot represents the
calibration behavior of the model for a specific subgroup, allowing for detailed
insight into how predicted probabilities align with actual outcomes across
demographic categories.

This type of disaggregated visualization is especially useful for fairness
analysis and subgroup performance auditing. By setting ``group_category="race"``,
the function automatically detects unique values in the specified column and
generates a separate calibration curve for each.

The dashed diagonal reference line represents perfect calibration. Brier scores
are included in each plot to provide a quantitative measure of calibration
performance within the group.

.. note::

    When using ``group_category``, both ``overlay`` and ``grid`` must be set to
    ``False``. This ensures each group receives its own standalone figure, avoiding
    conflicting layout behavior.

.. code-block:: python

    from model_metrics import show_calibration_curve

    show_calibration_curve(
        model=model_rf["model"].estimator,
        X=X_test,
        y=y_test,
        model_title="Random Forest Classifier",
        bins=10,
        show_brier_score=True,
        linestyle_kwgs={"color": "black"},
        curve_kwgs={title: {"linewidth": 2} for title in model_titles},
        group_category=X_test_2["race"],
    )

**Output**

.. raw:: html

   <div class="no-click">


.. image:: ../assets/calibration_3.svg
    :alt: Calibration Curves Example 3
    :width: 900px
    :align: center


.. raw:: html

    <div style="height: 40px;"></div>


.. [1] Efron, B., Hastie, T., Johnstone, I., & Tibshirani, R. (2004). *Diabetes Dataset*. Scikit-learn. Derived from: Efron, B., et al. (2004). Least Angle Regression. The Annals of Statistics, 32(2), 407-499. `https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset <https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset>`_.
.. [2] Kohavi, R. (1996). *Census Income*. UCI Machine Learning Repository. `https://doi.org/10.24432/C5GP7S <https://doi.org/10.24432/C5GP7S>`_.
.. [3] Funnell, A., Shpaner, L., & Petousis, P. (2024). *Model Tuner* (Version 0.0.28b) [Software]. Zenodo. `https://doi.org/10.5281/zenodo.12727322 <https://doi.org/10.5281/zenodo.12727322>`_.
