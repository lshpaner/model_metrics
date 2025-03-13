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

Model Training Example: Adult Census Income Dataset
----------------------------------------------------

.. code-block:: python

    import numpy as np

    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier

    ################################################################################
    ############################# Path Variables ###################################
    ################################################################################

    model_output = "model_output"  # model output path

    ################################################################################
    ############################ Global Constants ##################################
    ################################################################################

    rstate = 222  # random state for reproducibility

    ################################################################################
    ############################# Stratification ###################################
    ################################################################################

    # create bins for age along with labels such that age as a continuous series
    # can be converted to something more manageable for visualization and analysis

    bin_ages = [0, 18, 30, 40, 50, 60, 70, 80, 90, 100]

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
