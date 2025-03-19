
.. _caveats:

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


.. _threshold_selection_logic:

Threshold Selection logic
-------------------------------

When computing confusion matrices, selecting the right classification threshold 
can significantly impact the output. The function ``show_confusion_matrix`` is 
documented in :ref:`this section <confusion_matrix_evaluation>`.

1. If the custom_threshold parameter is passed, it takes absolute precedence and 
is used directly.

2. If ``model_threshold`` is set and the model contains a threshold dictionary, 
the function will try to retrieve the threshold using the score parameter:

- If ``score`` is passed (e.g., ``"f1"``), then ``model.threshold[score]`` is used.
- If ``score`` is not passed, the function will look up the first item in ``model.scoring`` (if available).
- If neither a custom threshold nor a valid model threshold is available, the default value of ``0.5`` is used.

.. _caveats_in_calibration:

Calibration Trade-offs
-----------------------------------------

Calibration curves are powerful diagnostic tools for assessing how well a model's 
predicted probabilities reflect actual outcomes. However, their interpretation—and 
the methods used to derive them—come with important caveats that users should keep 
in mind.

Calibration Methodology
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The examples shown in this library are based on models calibrated using Platt 
Scaling, a post-processing technique that fits a sigmoid function to the model's
prediction scores. **Platt Scaling** assumes a parametric form:

.. math::

   P(y = 1 \mid f(x)) = \frac{1}{1 + \exp(A f(x) + B)}

where :math:`A` and :math:`B` are scalar parameters learned using a separate 
calibration dataset. This approach is computationally efficient and works well 
for models such as SVMs and Logistic Regression, where prediction scores are 
linearly separable or approximately log-odds in nature.

However, Platt Scaling may underperform when the relationship between raw scores 
and true probabilities is non-monotonic or highly irregular.

Alternative Calibration Methods
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

An alternative to Platt Scaling is **Isotonic Regression**, a non-parametric method 
that fits a monotonically increasing function to the model's prediction scores. 
It is particularly effective when the mapping between predicted probabilities and 
observed outcomes is complex or non-linear.

Mathematically, isotonic regression solves the following constrained optimization problem:

.. math::

   \min_{\hat{p}_1, \ldots, \hat{p}_N} \sum_{i=1}^{N} (y_i - \hat{p}_i)^2 \quad \text{subject to} \quad \hat{p}_1 \leq \hat{p}_2 \leq \cdots \leq \hat{p}_N

Here:

- :math:`y_i \in \{0, 1\}` are the true binary labels,
- :math:`\hat{p}_i` are the calibrated probabilities corresponding to the model's scores,
- and the constraint enforces monotonicity, preserving the order of the original prediction scores.

The solution is obtained using the Pool Adjacent Violators Algorithm (PAVA), an 
efficient method for enforcing monotonicity in a least-squares fit.

While Isotonic Regression is highly flexible and can model arbitrary step-like 
functions, this same flexibility increases the risk of overfitting, especially 
when the calibration dataset is small, imbalanced, or noisy. It may capture 
spurious fluctuations in the validation data rather than the true underlying 
relationship between scores and outcomes.

.. warning:: 
   Overfitting with isotonic regression can lead to miscalibration in deployment, 
   particularly if the validation set is not representative of the production environment.

.. note:: 
   This library does not perform calibration internally. Instead, users are 
   expected to calibrate models during training or preprocessing—e.g., using the 
   ``model_tuner`` library [1]_ or any external tool. All calibration curve plots 
   included here are illustrative and assume models have already been calibrated 
   using Platt Scaling prior to visualization.

Dependence on Validation Data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
All calibration techniques rely heavily on the quality of the validation data 
used to learn the mapping. If the validation set is not representative of the 
target population, the resulting calibration curve may be misleading. This 
concern is especially important when deploying models in real-world settings 
where data drift or population imbalance may occur.

Interpreting the Brier Score
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The **Brier Score**, often reported alongside calibration curves, provides a 
quantitative measure of probabilistic accuracy. It is defined as:

.. math::

   \text{Brier Score} = \frac{1}{N} \sum_{i=1}^{N} (\hat{p}_i - y_i)^2

where :math:`\hat{p}_i` is the predicted probability and :math:`y_i` is the 
actual class label. While a lower Brier Score generally indicates better 
performance, it conflates calibration (how close predicted probabilities are to 
actual outcomes) and refinement (how confidently predictions are made). Thus, 
the Brier Score should be interpreted in context and not relied upon in isolation.

.. [1] Funnell, A., Shpaner, L., & Petousis, P. (2024). *Model Tuner* (Version 0.0.28b) [Software]. Zenodo. `https://doi.org/10.5281/zenodo.12727322 <https://doi.org/10.5281/zenodo.12727322>`_.
