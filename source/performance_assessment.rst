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
    :type model_titles: str or list, optional
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


ROC AUC Curves
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

The ``show_roc_curve`` function provides a flexible and powerful way to visualize 
the performance of binary classification models using Receiver Operating Characteristic 
(ROC) curves. Whether you're comparing multiple models, evaluating subgroup fairness, 
or preparing publication-ready plots, this function allows full control over layout,
styling, and annotations. It supports single and multiple model inputs, optional overlay 
or grid layouts, and group-wise comparisons via a categorical feature. Additional options 
allow custom axis labels, AUC precision, curve styling, and export to PNG/SVG. 
Designed to be both user-friendly and highly configurable, ``show_roc_curve`` 
is a practical tool for model evaluation and stakeholder communication.


.. function:: show_roc_curve(models, X, y, xlabel="False Positive Rate", ylabel="True Positive Rate", model_titles=None, decimal_places=2, overlay=False, title=None, save_plot=False, image_path_png=None, image_path_svg=None, text_wrap=None, curve_kwgs=None, linestyle_kwgs=None, grid=False, n_rows=None, n_cols=2, figsize=(8, 6), label_fontsize=12, tick_fontsize=10, gridlines=True, group_category=None)

    :param models: A trained model, a string placeholder, or a list containing models or strings to evaluate.
    :type models: object or str or list[object or str]
    :param X: Feature matrix used for prediction.
    :type X: pd.DataFrame or np.ndarray
    :param y: True binary labels for evaluation.
    :type y: pd.Series or np.ndarray
    :param xlabel: Label for the x-axis. Defaults to ``"False Positive Rate"``.
    :type xlabel: str, optional
    :param ylabel: Label for the y-axis. Defaults to ``"True Positive Rate"``.
    :type ylabel: str, optional
    :param model_titles: Custom titles for the models. Can be a string or list of strings. If ``None``, defaults to ``"Model 1"``, ``"Model 2"``, etc.
    :type model_titles: str or list[str], optional
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
    :param curve_kwgs: Plot styling for ROC curves. Accepts a list of dictionaries or a nested dictionary keyed by model_titles.
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
        - ``models`` and ``model_titles`` can be individual items or lists. Strings passed in ``models`` are treated as placeholder names.
        - Titles can be automatically inferred or explicitly passed using ``model_titles``.

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


ROC AUC Example 1 (Original)
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
        models=[model1, model2],
        X=X_test,
        y=y_test,
        model_titles=model_titles,
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
        models=model_dt["model"].estimator,
        X=X_test,
        y=y_test,
        model_titles="Decision Tree Classifier,
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


.. function:: show_pr_curve(models, X, y, xlabel="Recall", ylabel="Precision", model_titles=None, decimal_places=2, overlay=False, title=None, save_plot=False, image_path_png=None, image_path_svg=None, text_wrap=None, curve_kwgs=None, grid=False, n_rows=None, n_cols=2, figsize=(8, 6), label_fontsize=12, tick_fontsize=10, gridlines=True, group_category=None)

    :param models: A trained model, a string placeholder, or a list containing models or strings to evaluate.
    :type models: object or str or list[object or str]
    :param X: Feature matrix used for prediction.
    :type X: pd.DataFrame or np.ndarray
    :param y: True binary labels for evaluation.
    :type y: pd.Series or np.ndarray
    :param xlabel: Label for the x-axis. Defaults to ``"Recall"``.
    :type xlabel: str, optional
    :param ylabel: Label for the y-axis. Defaults to ``"Precision"``.
    :type ylabel: str, optional
    :param model_titles: Custom titles for the models. Can be a string or list of strings. If ``None``, defaults to ``"Model 1"``, ``"Model 2"``, etc.
    :type model_titles: str or list[str], optional
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
    :param curve_kwgs: Plot styling for PR curves. Accepts a list of dictionaries or a nested dictionary keyed by model_titles.
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
        - If ``model_titles`` is not a string, list of strings, or ``None``.

.. admonition:: Notes

    - **Flexible Inputs:**
        - ``models`` and ``model_titles`` can be individual items or lists. Strings passed in ``models`` are treated as placeholder names.
        - Titles can be automatically inferred or explicitly passed using ``model_titles``.

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


Precision-Recall Example 1 (Original)
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
        models=[logistic_model, rf_model],
        X=X_test,
        y=y_test,
        model_titles=["Logistic Regression", "Random Forest"],
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
        models=[model1, model2],
        X=X_test,
        y=y_test,
        model_titles=model_titles,
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
        models=model_dt["model"].estimator,
        X=X_test,
        y=y_test,
        model_titles="Decision Tree Classifier,
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










.. [1] Efron, B., Hastie, T., Johnstone, I., & Tibshirani, R. (2004). *Diabetes Dataset*. Scikit-learn. Derived from: Efron, B., et al. (2004). Least Angle Regression. The Annals of Statistics, 32(2), 407-499. `https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset <https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset>`_.
.. [2] Kohavi, R. (1996). *Census Income*. UCI Machine Learning Repository. `https://doi.org/10.24432/C5GP7S <https://doi.org/10.24432/C5GP7S>`_.
.. [3] Funnell, A., Shpaner, L., & Petousis, P. (2024). *Model Tuner* (Version 0.0.28b) [Software]. Zenodo. `https://doi.org/10.5281/zenodo.12727322 <https://doi.org/10.5281/zenodo.12727322>`_.
