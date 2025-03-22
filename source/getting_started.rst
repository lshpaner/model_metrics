.. _getting_started:   

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


Welcome to the Model Metrics Python Library Documentation!
============================================================
.. note::
   This documentation is for ``model_metrics`` version ``0.0.3a``.


Welcome to Model Metrics! Model Metrics is a versatile Python library designed 
to streamline the evaluation and interpretation of machine learning models. It 
provides a robust framework for generating predictions, computing model metrics, 
analyzing feature importance, and visualizing results. Whether you're working 
with SHAP values, model coefficients, confusion matrices, ROC curves, 
precision-recall plots, and other key performance indicators.


Project Links
---------------

1. `PyPI Page <https://pypi.org/project/model_metrics/>`_  

2. `GitHub Repository <https://github.com/lshpaner/model_metrics>`_


What is Model Evaluation?
-------------------------

Model evaluation is a fundamental aspect of the machine learning lifecycle. 
It involves assessing the performance of predictive models using various 
metrics to ensure accuracy, reliability, and fairness. Proper evaluation 
helps in understanding how well a model generalizes to unseen data, detects 
potential biases, and optimizes performance. This step is critical before 
deploying any model into production.


Purpose of Model Metrics Library
--------------------------------
The ``model_metrics`` library is a robust framework designed to simplify and 
standardize the evaluation of machine learning models. It provides an extensive 
set of tools to assess model performance, compare different approaches, and 
validate results with statistical rigor. Key functionalities include:

- **Performance Metrics:** A suite of functions to compute essential metrics such as accuracy, precision, recall, F1-score, ROC-AUC, log loss, and RMSE, among others.
- **Custom Evaluation Pipelines:** Predefined and customizable pipelines for automating model evaluation workflows.
- **Visualization Tools:** Functions to generate confusion matrices, ROC curves, precision-recall curves, calibration plots, and lift and gain charts.
- **Comparison and Benchmarking:** Frameworks to compare multiple models based on key metrics and statistical significance tests.


Key Features
------------

- **Comprehensive Evaluation:** Supports a wide range of model evaluation methods, ensuring a holistic assessment of predictive performance.  
- **User-Friendly:** Designed for ease of use, with intuitive functions and well-documented workflows.  
- **Customizable and Extensible:** Allows users to configure metric calculations and integrate with different machine learning frameworks.  
- **Seamless Integration:** Works with popular libraries such as ``Scikit-Learn``, ``XGBoost``, ``LightGBM``, and ``TensorFlow``, provided that model objects follow standard prediction interfaces like ``predict()``, ``predict_proba()``, or ``decision_function()``. Special considerations may be required for deep learning models, time-series models, or custom transformers that return non-standard outputs.  
- **Detailed Reports:** Provides automated summaries and visual insights to aid in model selection and decision-making.

.. _prerequisites:   

Prerequisites
-------------
Before you install ``model_metrics``, ensure your system meets the following requirements:

- **Python**: version ``3.7.4`` or higher is required to run ``model_metrics``.

Additionally, ``model_metrics`` depends on the following packages, which will be automatically installed when you install ``model_metrics``:

- ``matplotlib``: version ``3.5.3`` or higher, but capped at ``3.9.2``
- ``numpy``: version ``1.21.6`` or higher, but capped at ``2.1.0``
- ``pandas``: version ``1.3.5`` or higher, but capped at ``2.2.3``
- ``plotly``: version ``5.18.0`` or higher, but capped at ``5.24.0``
- ``scikit-learn``: version ``1.0.2`` or higher, but capped at ``1.5.2``
- ``shap``: version ``0.41.0`` or higher, but capped below ``0.46.0``
- ``statsmodels``: version ``0.12.2`` or higher, but capped below ``0.14.4``
- ``tqdm```: version ``4.66.4`` or higher, but capped below ``4.67.1``

.. _installation:

Installation
-------------

You can install ``model_metrics`` directly from PyPI:

.. code-block:: python

    pip install model_metrics

Description
--------------

This guide provides detailed instructions and examples for using the functions 
provided in the ``model_metrics`` library and how to use them effectively in your projects.

For most of the ensuing examples, we will leverage the Census Income Data (1994) from
the UCI Machine Learning Repository [#]_. This dataset provides a rich source of
information for demonstrating the functionalities of the ``model_metrics``.

.. [#] Kohavi, R. (1996). *Census Income*. UCI Machine Learning Repository. `https://doi.org/10.24432/C5GP7S <https://doi.org/10.24432/C5GP7S>`_.