
.. _model_training:

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


.. _Adult_Income_Training:

Adult Income Dataset
=============================

This example shows an example of what a ``train.py`` script could look like to train
and concisely evaluate classification models on the Adult Census Income dataset [1]_ using
the ``model_tuner`` library [2]_. The dataset presents a classic binary classification 
problem—predicting whether an individual's income exceeds $50K per year—based on 
a variety of demographic and employment-related features. The model_tuner 
framework streamlines model development by handling preprocessing, cross-validation, 
hyperparameter tuning, and performance evaluation, enabling rapid experimentation 
with minimal boilerplate code.

Model Configuration & Hyperparameters (``model_params.py``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This script defines the model configurations and hyperparameter grids for three 
classifiers—Logistic Regression, Decision Tree, and Random Forest—used in our 
evaluation of the Adult Income dataset [1]_.

Each model is structured into a standardized dictionary format compatible with 
the model_tuner pipeline, containing the estimator, its name, a hyperparameter grid, 
and metadata flags for tuning strategies. To ensure consistent and reproducible results, 
a fixed random state (``rstate = 222``) is used across models.

The **Logistic Regression** model is tuned over a range of regularization strengths 
(``C``) using an ``L2`` penalty. It is configured with class balancing and parallel 
processing to improve performance and scalability. The **Decision Tree** model 
explores various tree depths along with different minimum sample thresholds for 
both splits and leaves, offering a balance between flexibility and regularization, 
with class balancing also enabled. Lastly, the **Random Forest** model is optimized 
for efficiency by using a reduced number of estimators and constraining tree depth 
to help prevent overfitting. Like Logistic Regression, it leverages parallel 
processing for faster computation.

All three model definitions are collected in a dictionary (``model_definitions``) that can be easily passed into the model_tuner workflow for training, evaluation, and comparison.

Let me know if you'd like to add default scoring metrics or cross-validation strategy info to this section as well.

.. code-block:: python

    import numpy as np

    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier

    ################################################################################
    ############################ Global Constants ##################################
    ################################################################################

    rstate = 222  # random state for reproducibility

    ################################################################################
    ########################## Logistic Regression #################################
    ################################################################################

    # Define the hyperparameters for Logistic Regression
    lr_name = "lr"

    lr_penalties = ["l2"]
    lr_Cs = np.logspace(-4, 0, 5)
    # lr_max_iter = [100, 500]

    # Structure the parameters similarly to the RF template
    tuned_parameters_lr = [
        {
            "lr__penalty": lr_penalties,
            "lr__C": lr_Cs,
        }
    ]

    lr = LogisticRegression(
        class_weight="balanced",
        random_state=rstate,
        n_jobs=2,
    )

    lr_definition = {
        "clc": lr,
        "estimator_name": lr_name,
        "tuned_parameters": tuned_parameters_lr,
        "randomized_grid": False,
        "early": False,
    }

    ################################################################################
    ############################### Decision Trees #################################
    ################################################################################

    # Define Decision Tree parameters
    dt_name = "dt"

    # Simplified hyperparameters
    dt_max_depth = [None, 10, 20]  # Unbounded, shallow, and medium depths
    dt_min_samples_split = [2, 10]  # Default and a stricter option
    dt_min_samples_leaf = [1, 5]  # Default and larger leaf nodes for regularization

    # Define the parameter grid for Decision Trees
    tuned_parameters_dt = [
        {
            "dt__max_depth": dt_max_depth,
            "dt__min_samples_split": dt_min_samples_split,
            "dt__min_samples_leaf": dt_min_samples_leaf,
        }
    ]

    # Define the Decision Tree model
    dt = DecisionTreeClassifier(
        class_weight="balanced",
        random_state=rstate,
    )

    # Define the Decision Tree model configuration
    dt_definition = {
        "clc": dt,
        "estimator_name": dt_name,
        "tuned_parameters": tuned_parameters_dt,
        "randomized_grid": False,
        "early": False,
    }

    ################################################################################
    ##############################  Random Forest  #################################
    ################################################################################


    # Define the hyperparameters for Random Forest (trimmed for efficiency)
    rf_name = "rf"

    # Reduced hyperparameters for tuning
    rf_parameters = [
        {
            "rf__n_estimators": [10, 50],  # Reduce number of trees for speed
            "rf__max_depth": [None, 10],  # Limit depth to prevent overfitting
            "rf__min_samples_split": [2, 5],  # Fewer options for splitting
        }
    ]

    # Initialize the Random Forest Classifier with a smaller number of trees
    rf = RandomForestClassifier(
        n_estimators=10,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )

    # Define the Random Forest model setup
    rf_definition = {
        "clc": rf,
        "estimator_name": rf_name,
        "tuned_parameters": rf_parameters,
        "randomized_grid": False,
        "early": False,
    }

    ################################################################################

    model_definitions = {
        lr_name: lr_definition,
        dt_name: dt_definition,
        rf_name: rf_definition,
    }


Model Training (``train.py``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This script defines the end-to-end workflow for training and evaluating 
classification models on the Adult Income dataset using the ``model_tuner`` library [2]_. 
The process is structured as a command-line interface (CLI) application using `typer <https://pypi.org/project/typer/>`_, 
with steps that include dataset fetching, preprocessing, model tuning, training, 
calibration, evaluation, and serialization.

After importing the necessary libraries and establishing paths, the script fetches 
the Adult dataset [1]_ directly from the UCI Mahcine Learning Repository [3]_ via 
``ucimlrepo``. The target column is cleaned and encoded into binary format, 
while only numeric features are retained for modeling. Additionally, subgroup 
columns (race and sex) are extracted for stratified sampling.

The script retrieves the model configuration based on the model_type argument 
(e.g., ``"lr"``, ``"dt"``, or ``"rf"``), then builds a preprocessing pipeline consisting of 
a standard scaler and simple imputer. This pipeline is passed into the Model 
class from ``model_tuner``, along with tuning parameters, model metadata, and 
training settings—including stratification, calibration, and scoring criteria.

A grid search is performed with optional F1-beta optimization, and the dataset 
is split into training, validation, and test sets. The selected model is trained, 
optionally calibrated, and evaluated using ROC AUC as the primary metric. Final 
metrics are printed and returned as a DataFrame, and the trained model object is 
saved to disk for future use.

.. code-block:: python

    ################################################################################
    ## Step 1. Import Libraries
    ################################################################################

    from pathlib import Path
    import typer
    from loguru import logger
    import pandas as pd
    import numpy as np
    from ucimlrepo import fetch_ucirepo
    import os
    import model_tuner
    from model_tuner import Model, dumpObjects
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from py_scripts.model_params import model_definitions, rstate

    ################################################################################
    ## Step 2. Initialize CLI App and Paths
    ################################################################################

    app = typer.Typer()

    PROCESSED_DATA_DIR = Path("model_files")
    MODELS_DIR = Path("model_files")
    RESULTS_DIR = Path("model_files/results")

    ################################################################################
    ## Step 3. Define Main Function with CLI Command
    ################################################################################


    @app.command()
    def main(
        model_type: str = "lr",
    ):

        ############################################################################
        ## Display Model Tuner Version Info
        ############################################################################

        print(f"\nModel Tuner version: {model_tuner.__version__}\n")
        print(f"Model Tuner authors: {model_tuner.__author__}\n")

        ############################################################################
        ## Step 4. Fetch and Prepare Dataset
        ############################################################################

        # Fetch dataset
        adult = fetch_ucirepo(id=2)  # UCI Adult dataset
        X = adult.data.features
        y = adult.data.targets

        # Copy X to retrieve original features, used for stratification
        stratify_df = X.copy()

        # Log first five rows of features and targets
        logger.info(f"\n{'=' * 80}\nX\n{'=' * 80}\n{X.head()}")
        logger.info(f"\n{'=' * 80}\ny\n{'=' * 80}\n{y.head()}")

        # Retain numeric columns only
        X = X.select_dtypes(include=np.number)

        # Subset stratify_df to those features uses for stratification
        stratify_df = stratify_df[["race", "sex"]]

        # Clean target column by removing trailing period
        y.loc[:, "income"] = y["income"].str.rstrip(".")

        # Display class balance
        print(f"\nBreakdown of y:\n{y['income'].value_counts()}\n")

        # Encode target to binary
        y = y["income"].map({"<=50K": 0, ">50K": 1})

        ############################################################################
        ## Step 5. Extract Model Settings
        ############################################################################

        clc = model_definitions[model_type]["clc"]
        estimator_name = model_definitions[model_type]["estimator_name"]

        # Set the parameters
        tuned_parameters = model_definitions[model_type]["tuned_parameters"]
        early_stop = model_definitions[model_type]["early"]

        metrics = {}

        logger.info(f"\nTraining {estimator_name}...")

        ############################################################################
        ## Step 6. Create Preprocessing Pipeline
        ###########################################################################

        pipeline = [
            ("StandardScalar", StandardScaler()),
            ("Preprocessor", SimpleImputer()),
        ]

        print("\n" + "=" * 60)

        ############################################################################
        ## Step 7. Instantiate Model Tuner
        ############################################################################

        model = Model(
            pipeline_steps=pipeline,
            name=estimator_name,
            model_type="classification",
            estimator_name=estimator_name,
            calibrate=True,
            estimator=clc,
            kfold=False,
            grid=tuned_parameters,
            n_jobs=2,
            randomized_grid=False,
            scoring=["roc_auc"],
            random_state=rstate,
            stratify_cols=stratify_df,
            stratify_y=True,
            boost_early=early_stop,
        )

        ############################################################################
        ## Step 8. Grid Search & Data Splitting
        ############################################################################

        model.grid_search_param_tuning(X, y, f1_beta_tune=True)
        X_test, y_test = model.get_test_data(X, y)
        X_valid, y_valid = model.get_valid_data(X, y)

        ############################################################################
        ## Step 9. Train and Calibrate Model
        ############################################################################

        model.fit(X, y, score="roc_auc")

        if model.calibrate:
            model.calibrateModel(X, y, score="roc_auc")

        ############################################################################
        ## Step 10. Evaluate Model
        ############################################################################

        return_metrics_dict = model.return_metrics(
            X,
            y,
            optimal_threshold=True,
            print_threshold=True,
            model_metrics=True,
            return_dict=True,
        )

        metrics = pd.Series(return_metrics_dict).to_frame(estimator_name)
        metrics = round(metrics, 3)
        print("=" * 80)

        ############################################################################
        ## Step 11. Save Trained Model
        ############################################################################

        print("=" * 80)
        dumpObjects(
            {
                "model": model,
            },
            RESULTS_DIR / f"{str(clc).split('(')[0]}.pkl",
        )


    if __name__ == "__main__":
        app()

Loading (Retrieving) The Model Objects and Data Splits
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    model_path = RESULTS_DIR
    
    model_lr = loadObjects(os.path.join(model_path, "LogisticRegression.pkl"))
    model_dt = loadObjects(os.path.join(model_path, "DecisionTreeClassifier.pkl"))
    model_rf = loadObjects(os.path.join(model_path, "RandomForestClassifier.pkl"))


    X_test = pd.read_parquet(os.path.join(data_path, "X_test.parquet"))
    y_test = pd.read_parquet(os.path.join(data_path, "y_test.parquet"))



.. [1] Kohavi, R. (1996). *Census Income*. UCI Machine Learning Repository. `https://doi.org/10.24432/C5GP7S <https://doi.org/10.24432/C5GP7S>`_.
.. [2] Funnell, A., Shpaner, L., & Petousis, P. (2024). *Model Tuner* (Version 0.0.28b) [Software]. Zenodo. `https://doi.org/10.5281/zenodo.12727322 <https://doi.org/10.5281/zenodo.12727322>`_.
.. [3] Dua, D. & Graff, C. (2019). UCI Machine Learning Repository. University of California, Irvine, School of Information and Computer Sciences. `https://archive.ics.uci.edu <https://archive.ics.uci.edu>`_.
