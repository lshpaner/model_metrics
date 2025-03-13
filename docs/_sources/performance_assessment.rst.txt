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

.. function:: summarize_model_performance(model, X, y, model_type="classification", model_threshold=None, model_titles=None, custom_threshold=None, score=None, return_df=False, overall_only=False, decimal_places=3)

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
    :param model_titles: Custom model names for display. If ``None``, names are inferred from the models. Defaults to ``None``.
    :type model_titles: list, optional
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

    model_titles = ["Logistic Regression", "Random Forest"]



Binary Classification Example 1
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from model_metrics import summarize_model_performance

    model_performance = summarize_model_performance(
        model=[model1, model2],
        model_titles=model_titles,
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
        model_titles=model_titles,
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
        model_titles=["Linear Regression", "Ridge Regression"],
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
        model_titles=["Linear Regression", "Ridge Regression", "Random Forest"],
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
        model_titles=["Linear Regression", "Ridge Regression", "Random Forest"],
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


ROC AUC Evaluation
--------------------

This section demonstrates how to evaluate the performance of binary classification 
models using ROC AUC curves, a key metric for assessing the trade-off between 
true positive and false positive rates. Using the Logistic Regression and 
Random Forest Classifier models trained on the :ref:`synthetic dataset from the 
previous (Binary Classification Models) section <Binary_Classification>`, 
we generate ROC curves to visualize their discriminatory power.

ROC AUC (Receiver Operating Characteristic Area Under the Curve) provides a 
single scalar value representing a model’s ability to distinguish between 
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

ROC AUC Evaluation Example 1
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
        models=[model1, model2],
        X=X_test,
        y=y_test,
        model_titles=model_titles,
        decimal_places=2,
        n_cols=2,
        n_rows=1,
        curve_kwgs={
            "Logistic Regression": {"color": "blue", "linewidth": 2},
            "Random Forest": {"color": "green", "linewidth": 2},
        },
        linestyle_kwgs={"color": "red", "linestyle": "--"},
        save_plot=True,
        grid=True,
        figsize=(12, 6),
    )


.. raw:: html

   <div class="no-click">

.. image:: ../assets/grid_roc_auc.svg
   :alt: ROC AUC Example 1
   :align: left
   :width: 900px

.. raw:: html

   </div>


ROC AUC Evaluation Example 2
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In this second ROC AUC evaluation example, we focus on overlaying the results of 
two models—Logistic Regression and Random Forest Classifier—trained on the 
:ref:`synthetic dataset from the Binary Classification Models section 
<Binary_Classification>` onto a single plot. Using the ``show_roc_curve`` 
function with the ``overlay=True`` parameter, the ROC curves for both models are 
displayed together, with Logistic Regression in blue and Random Forest in black, 
both with a ``linewidth=2``. A red dashed line serves as the random guessing 
baseline, and the plot includes a custom title for clarity.



.. raw:: html

   <div class="no-click">

.. image:: ../assets/overlay_roc_auc_plot.svg
   :alt: ROC AUC Example 1
   :align: left
   :width: 900px

.. raw:: html

   </div>


ROC AUC Evaluation Example 3
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In this third ROC AUC evaluation example, we utilize the well-known Adult Income 
dataset [2]_, a practical benchmark for binary classification tasks. This dataset 
contains a rich set of categorical and numerical features that make it particularly 
suitable for analyzing model performance across different subgroups.

To build and evaluate our model, we make use of the ``model_tuner`` library [3]_. 
:ref:`Click here to view the corresponding codebase for this workflow. <Adult_Income_Training>`

In this scenario, our objective is to assess ROC AUC scores **not just overall**, 
but **for each individual category within a selected feature**—such as *occupation, 
education,* or *marital-status*. This enables a deeper examination of how model 
performance may vary across groups, which is especially important in contexts 
involving fairness, bias detection, or subgroup-level interpretability.

Using the ``show_roc_curve`` function, we compute and plot the ROC AUC scores 
for a model (e.g., Random Forest) across all unique values in the selected feature. 
For example, when analyzing the *occupation* column, the plot shows the ROC AUC for 
each job category, helping us understand whether the classifier performs consistently 
across different employment types.

The resulting visualization displays all ROC curves on a shared axis, enabling 
direct comparison across categories. Customization options include curve color, 
line style, plot size, and title, giving users flexibility to tailor the output 
to their analytical needs. This subgroup-level visualization provides a powerful 
lens for identifying discrepancies in model behavior—especially in scenarios 
where transparency, fairness, and accountability are key.


.. raw:: html

    <div style="height: 40px;"></div>

.. [1] Efron, B., Hastie, T., Johnstone, I., & Tibshirani, R. (2004). *Diabetes Dataset*. Scikit-learn. Derived from: Efron, B., et al. (2004). Least Angle Regression. The Annals of Statistics, 32(2), 407-499. `https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset <https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset>`_.
.. [2] Kohavi, R. (1996). *Census Income*. UCI Machine Learning Repository. `https://doi.org/10.24432/C5GP7S <https://doi.org/10.24432/C5GP7S>`_.
.. [3] Funnell, A., Shpaner, L., & Petousis, P. (2024). *Model Tuner* (Version 0.0.28b) [Software]. Zenodo. `https://doi.org/10.5281/zenodo.12727322 <https://doi.org/10.5281/zenodo.12727322>`_.
