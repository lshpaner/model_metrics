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

Model Performance Summary
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

    - If ``return_df=False``, the function prints results in a structured format with center-aligned headers.

The ``summarize_model_performance`` function provides a structured evaluation of classification and regression models, generating key performance metrics. For classification models, it computes precision, recall, specificity, F1-score, and AUC ROC. For regression models, it extracts coefficients and evaluates error metrics like MSE, RMSE, and R². The function allows specifying custom thresholds, metric rounding, and formatted display options.

Binary Classification Example
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Below is an example demonstrating how to evaluate multiple models using ``summarize_model_performance``. The function calculates and presents metrics for classification and regression models.

.. code-block:: python

    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from model_metrics import summarize_model_performance

    # Generate a synthetic dataset
    X, y = make_classification(n_samples=1000, n_features=10, random_state=42)

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train models
    model1 = LogisticRegression().fit(X_train, y_train)
    model2 = RandomForestClassifier().fit(X_train, y_train)

    model_titles = ["Logistic Regression", "Random Forest"]

    # Evaluate model performance
    model_performance = summarize_model_performance(
        model=[model1, model2],
        model_titles=model_titles,
        X=X_test,
        y=y_test,
        model_type="classification",
        return_df=True
    )

    model_performance

**Output**

.. raw:: html

    <style type="text/css">
    .tg  {border-collapse:collapse;border-spacing:0;}
    .tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:12px;
    overflow:hidden;padding:10px 5px;word-break:normal;}
    .tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:12px;
    font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
    .tg .tg-7btt{border-color:inherit;font-weight:bold;text-align:center;vertical-align:top}
    .tg .tg-0pky{border-color:inherit;text-align:center;vertical-align:top}
    </style>
    <table class="tg"><thead>
    <tr>
        <th class="tg-7btt">Model</th>
        <th class="tg-7btt">Logistic Regression</th>
        <th class="tg-7btt">Random Forest Classifier</th>
    </tr></thead>
    <tbody>
    <tr>
        <td class="tg-0pky">Precision/PPV</td>
        <td class="tg-0pky">0.867</td>
        <td class="tg-0pky">0.914</td>
    </tr>
    <tr>
        <td class="tg-0pky">Average Precision</td>
        <td class="tg-0pky">0.937</td>
        <td class="tg-0pky">0.967</td>
    </tr>
    <tr>
        <td class="tg-0pky">Sensitivity/Recall</td>
        <td class="tg-0pky">0.820</td>
        <td class="tg-0pky">0.865</td>
    </tr>
    <tr>
        <td class="tg-0pky">Specificity</td>
        <td class="tg-0pky">0.843</td>
        <td class="tg-0pky">0.899</td>
    </tr>
    <tr>
        <td class="tg-0pky">F1-Score</td>
        <td class="tg-0pky">0.843</td>
        <td class="tg-0pky">0.889</td>
    </tr>
    <tr>
        <td class="tg-0pky">AUC   ROC</td>
        <td class="tg-0pky">0.913</td>
        <td class="tg-0pky">0.950</td>
    </tr>
    </tbody>
    </table>

.. raw:: html

    <div style="height: 40px;"></div>


Regression Example
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from sklearn.datasets import make_regression
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from model_metrics import summarize_model_performance

    # Generate a synthetic regression dataset
    X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train regression models
    model1 = LinearRegression().fit(X_train, y_train)
    model2 = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)

    # Evaluate regression model performance
    # Evaluate models
    regression_metrics = summarize_model_performance(
        model=[linear_model, rf_model],
        model_titles=["Linear Regression", "Random Forest"],
        X=X_test_scaled,
        y=y_test,
        model_type="regression",
        return_df=True,
    )

    regression_metrics

.. raw:: html

    <style type="text/css">
    .tg {
        border-collapse: collapse;
        border-spacing: 0;
        max-width: 450px; 
        width: 100%;
    }
    .tg td {
        border-color: black;
        border-style: solid;
        border-width: 1px;
        font-family: Arial, sans-serif;
        font-size: 12px; 
        overflow: hidden;
        padding: 0px 3px; 
        word-break: normal;
    }
    .tg th {
        border-color: black;
        border-style: solid;
        border-width: 1px;
        font-family: Arial, sans-serif;
        font-size: 12px; 
        font-weight: normal;
        overflow: hidden;
        padding: 0px 3px; 
        word-break: normal;
    }
    .tg .tg-2b7s { text-align: right; vertical-align: bottom; }
    .tg .tg-kex3 { font-weight: bold; text-align: right; vertical-align: bottom; }
    .tg .tg-j6zm { font-weight: bold; text-align: right; vertical-align: bottom; }
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
                <col style="width: 106px"> <!-- Model -->
                <col style="width: 67px"> <!-- Metric -->
                <col style="width: 52px"> <!-- Variable -->
                <col style="width: 70px"> <!-- Coefficient -->
                <col style="width: 50px"> <!-- P-value -->
                <col style="width: 45px"> <!-- MAE -->
                <col style="width: 70px"> <!-- MAPE (%) -->
                <col style="width: 55px"> <!-- MSE -->
                <col style="width: 45px"> <!-- RMSE -->
                <col style="width: 65px"> <!-- Expl. Var. -->
                <col style="width: 70px"> <!-- R^2 Score -->
            </colgroup>
            <thead>
                <tr>
                    <th class="tg-kex3">Model</th>
                    <th class="tg-kex3">Metric</th>
                    <th class="tg-kex3">Variable</th>
                    <th class="tg-kex3">Coefficient</th>
                    <th class="tg-kex3">P-value</th>
                    <th class="tg-kex3">MAE</th>
                    <th class="tg-kex3">MAPE (%)</th>
                    <th class="tg-kex3">MSE</th>
                    <th class="tg-kex3">RMSE</th>
                    <th class="tg-kex3">Expl. Var.</th>
                    <th class="tg-j6zm">R^2 Score</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td class="tg-2b7s">Linear Regression</td>
                    <td class="tg-2b7s">Overall Metrics</td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s">42.794</td>
                    <td class="tg-2b7s">37.5</td>
                    <td class="tg-2b7s">2900.194</td>
                    <td class="tg-2b7s">53.853</td>
                    <td class="tg-2b7s">0.455</td>
                    <td class="tg-7zrl">0.453</td>
                </tr>
                <tr>
                    <td class="tg-2b7s">Linear Regression</td>
                    <td class="tg-2b7s">Coefficient</td>
                    <td class="tg-2b7s">0</td>
                    <td class="tg-2b7s">156.463</td>
                    <td class="tg-2b7s">0</td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-7zrl"></td>
                </tr>
                <tr>
                    <td class="tg-2b7s">Linear Regression</td>
                    <td class="tg-2b7s">Coefficient</td>
                    <td class="tg-2b7s">1</td>
                    <td class="tg-2b7s">-6.258</td>
                    <td class="tg-2b7s">0.285</td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-7zrl"></td>
                </tr>
                <tr>
                    <td class="tg-2b7s">Linear Regression</td>
                    <td class="tg-2b7s">Coefficient</td>
                    <td class="tg-2b7s">2</td>
                    <td class="tg-2b7s">-12.247</td>
                    <td class="tg-2b7s">0.121</td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-7zrl"></td>
                </tr>
                <tr>
                    <td class="tg-2b7s">Linear Regression</td>
                    <td class="tg-2b7s">Coefficient</td>
                    <td class="tg-2b7s">3</td>
                    <td class="tg-2b7s">20.091</td>
                    <td class="tg-2b7s">0.004</td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-7zrl"></td>
                </tr>
                <tr>
                    <td class="tg-2b7s">Linear Regression</td>
                    <td class="tg-2b7s">Coefficient</td>
                    <td class="tg-2b7s">4</td>
                    <td class="tg-2b7s">10.902</td>
                    <td class="tg-2b7s">0.219</td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-7zrl"></td>
                </tr>
                <tr>
                    <td class="tg-2b7s">Linear Regression</td>
                    <td class="tg-2b7s">Coefficient</td>
                    <td class="tg-2b7s">5</td>
                    <td class="tg-2b7s">16.991</td>
                    <td class="tg-2b7s">0.8</td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-7zrl"></td>
                </tr>
                <tr>
                    <td class="tg-2b7s">Linear Regression</td>
                    <td class="tg-2b7s">Coefficient</td>
                    <td class="tg-2b7s">6</td>
                    <td class="tg-2b7s">-11.763</td>
                    <td class="tg-2b7s">0.838</td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-7zrl"></td>
                </tr>
                <tr>
                    <td class="tg-2b7s">Linear Regression</td>
                    <td class="tg-2b7s">Coefficient</td>
                    <td class="tg-2b7s">7</td>
                    <td class="tg-2b7s">-19.65</td>
                    <td class="tg-2b7s">0.503</td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-7zrl"></td>
                </tr>
                <tr>
                    <td class="tg-2b7s">Linear Regression</td>
                    <td class="tg-2b7s">Coefficient</td>
                    <td class="tg-2b7s">8</td>
                    <td class="tg-2b7s">-8.769</td>
                    <td class="tg-2b7s">0.606</td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-7zrl"></td>
                </tr>
                <tr>
                    <td class="tg-2b7s">Linear Regression</td>
                    <td class="tg-2b7s">Coefficient</td>
                    <td class="tg-2b7s">9</td>
                    <td class="tg-2b7s">28.723</td>
                    <td class="tg-2b7s">0.224</td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-7zrl"></td>
                </tr>
                <tr>
                    <td class="tg-2b7s">Linear Regression</td>
                    <td class="tg-2b7s">Coefficient</td>
                    <td class="tg-2b7s">10</td>
                    <td class="tg-2b7s">7.99</td>
                    <td class="tg-2b7s">0.317</td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-7zrl"></td>
                </tr>
                <tr>
                    <td class="tg-2b7s">Random Forest</td>
                    <td class="tg-2b7s">Overall Metrics</td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s">44.171</td>
                    <td class="tg-2b7s">40.09</td>
                    <td class="tg-2b7s">2959.181</td>
                    <td class="tg-2b7s">54.398</td>
                    <td class="tg-2b7s">0.442</td>
                    <td class="tg-7zrl">0.441</td>
                </tr>
                <tr>
                    <td class="tg-2b7s">Random Forest</td>
                    <td class="tg-2b7s">Coefficient</td>
                    <td class="tg-2b7s">0</td>
                    <td class="tg-2b7s">156.463</td>
                    <td class="tg-2b7s">0</td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-7zrl"></td>
                </tr>
                <tr>
                    <td class="tg-2b7s">Random Forest</td>
                    <td class="tg-2b7s">Coefficient</td>
                    <td class="tg-2b7s">1</td>
                    <td class="tg-2b7s">-6.258</td>
                    <td class="tg-2b7s">0.285</td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-7zrl"></td>
                </tr>
                <tr>
                    <td class="tg-2b7s">Random Forest</td>
                    <td class="tg-2b7s">Coefficient</td>
                    <td class="tg-2b7s">2</td>
                    <td class="tg-2b7s">-12.247</td>
                    <td class="tg-2b7s">0.121</td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-7zrl"></td>
                </tr>
                <tr>
                    <td class="tg-2b7s">Random Forest</td>
                    <td class="tg-2b7s">Coefficient</td>
                    <td class="tg-2b7s">3</td>
                    <td class="tg-2b7s">20.091</td>
                    <td class="tg-2b7s">0.004</td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-7zrl"></td>
                </tr>
                <tr>
                    <td class="tg-2b7s">Random Forest</td>
                    <td class="tg-2b7s">Coefficient</td>
                    <td class="tg-2b7s">4</td>
                    <td class="tg-2b7s">10.902</td>
                    <td class="tg-2b7s">0.219</td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-7zrl"></td>
                </tr>
                <tr>
                    <td class="tg-2b7s">Random Forest</td>
                    <td class="tg-2b7s">Coefficient</td>
                    <td class="tg-2b7s">5</td>
                    <td class="tg-2b7s">16.991</td>
                    <td class="tg-2b7s">0.8</td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-7zrl"></td>
                </tr>
                <tr>
                    <td class="tg-2b7s">Random Forest</td>
                    <td class="tg-2b7s">Coefficient</td>
                    <td class="tg-2b7s">6</td>
                    <td class="tg-2b7s">-11.763</td>
                    <td class="tg-2b7s">0.838</td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-7zrl"></td>
                </tr>
                <tr>
                    <td class="tg-2b7s">Random Forest</td>
                    <td class="tg-2b7s">Coefficient</td>
                    <td class="tg-2b7s">7</td>
                    <td class="tg-2b7s">-19.65</td>
                    <td class="tg-2b7s">0.503</td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-7zrl"></td>
                </tr>
                <tr>
                    <td class="tg-2b7s">Random Forest</td>
                    <td class="tg-2b7s">Coefficient</td>
                    <td class="tg-2b7s">8</td>
                    <td class="tg-2b7s">-8.769</td>
                    <td class="tg-2b7s">0.606</td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-7zrl"></td>
                </tr>
                <tr>
                    <td class="tg-2b7s">Random Forest</td>
                    <td class="tg-2b7s">Coefficient</td>
                    <td class="tg-2b7s">9</td>
                    <td class="tg-2b7s">28.723</td>
                    <td class="tg-2b7s">0.224</td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-7zrl"></td>
                </tr>
                <tr>
                    <td class="tg-2b7s">Random Forest</td>
                    <td class="tg-2b7s">Coefficient</td>
                    <td class="tg-2b7s">10</td>
                    <td class="tg-2b7s">7.99</td>
                    <td class="tg-2b7s">0.317</td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-2b7s"></td>
                    <td class="tg-7zrl"></td>
                </tr>
            </tbody>
        </table>
    </div>



.. raw:: html

    <div style="height: 40px;"></div>