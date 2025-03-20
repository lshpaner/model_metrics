
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


Partial Dependence Foundations
--------------------------------

Let :math:`\mathbf{X}` represent the complete set of input features for a machine 
learning model, where :math:`\mathbf{X} = \{X_1, X_2, \dots, X_p\}`. Suppose we're 
particularly interested in a subset of these features, denoted by :math:`\mathbf{X}_S`. 
The complementary set, :math:`\mathbf{X}_C`, contains all the features in :math:`\mathbf{X}` 
that are not in :math:`\mathbf{X}_S`. Mathematically, this relationship is expressed as:

.. math::

   \mathbf{X}_C = \mathbf{X} \setminus \mathbf{X}_S

where :math:`\mathbf{X}_C` is the set of features in :math:`\mathbf{X}` after 
removing the features in :math:`\mathbf{X}_S`.

Partial Dependence Plots (PDPs) are used to illustrate the effect of the features 
in :math:`\mathbf{X}_S` on the model's predictions, while averaging out the 
influence of the features in :math:`\mathbf{X}_C`. This is mathematically defined as:

.. math::
   \begin{align*}
   \text{PD}_{\mathbf{X}_S}(x_S) &= \mathbb{E}_{\mathbf{X}_C} \left[ f(x_S, \mathbf{X}_C) \right] \\
   &= \int f(x_S, x_C) \, p(x_C) \, dx_C \\
   &= \int \left( \int f(x_S, x_C) \, p(x_C \mid x_S) \, dx_C \right) p(x_S) \, dx_S
   \end{align*}


where:

- :math:`\mathbb{E}_{\mathbf{X}_C} \left[ \cdot \right]` indicates that we are taking the expected value over the possible values of the features in the set :math:`\mathbf{X}_C`.
- :math:`p(x_C)` represents the probability density function of the features in :math:`\mathbf{X}_C`.

This operation effectively summarizes the model's output over all potential values of the complementary features, providing a clear view of how the features in :math:`\mathbf{X}_S` alone impact the model's predictions.


**2D Partial Dependence Plots**

Consider a trained machine learning model :ref:`2D Partial Dependence Plots <2D_Partial_Dependence_Plots>`:math:`f(\mathbf{X})`, where :math:`\mathbf{X} = (X_1, X_2, \dots, X_p)` represents the vector of input features. The partial dependence of the predicted response :math:`\hat{y}` on a single feature :math:`X_j` is defined as:

.. math::

   \text{PD}(X_j) = \frac{1}{n} \sum_{i=1}^{n} f(X_j, \mathbf{X}_{C_i})

where:

- :math:`X_j` is the feature of interest.
- :math:`\mathbf{X}_{C_i}` represents the complement set of :math:`X_j`, meaning the remaining features in :math:`\mathbf{X}` not included in :math:`X_j` for the :math:`i`-th instance.
- :math:`n` is the number of observations in the dataset.

For two features, :math:`X_j` and :math:`X_k`, the partial dependence is given by:

.. math::

   \text{PD}(X_j, X_k) = \frac{1}{n} \sum_{i=1}^{n} f(X_j, X_k, \mathbf{X}_{C_i})

This results in a 2D surface plot (or contour plot) that shows how the predicted outcome changes as the values of :math:`X_j` and :math:`X_k` vary, while the effects of the other features are averaged out.

- **Single Feature PDP:** When plotting :math:`\text{PD}(X_j)`, the result is a 2D line plot showing the marginal effect of feature :math:`X_j` on the predicted outcome, averaged over all possible values of the other features.
- **Two Features PDP:** When plotting :math:`\text{PD}(X_j, X_k)`, the result is a 3D surface plot (or a contour plot) that shows the combined marginal effect of :math:`X_j` and :math:`X_k` on the predicted outcome. The surface represents the expected value of the prediction as :math:`X_j` and :math:`X_k` vary, while all other features are averaged out.


**3D Partial Dependence Plots**

For a more comprehensive analysis, especially when exploring interactions between two features, :ref:`3D Partial Dependence Plots <3D_Partial_Dependence_Plots>` are invaluable. The partial dependence function for two features in a 3D context is:

.. math::

   \text{PD}(X_j, X_k) = \frac{1}{n} \sum_{i=1}^{n} f(X_j, X_k, \mathbf{X}_{C_i})

Here, the function :math:`f(X_j, X_k, \mathbf{X}_{C_i})` is evaluated across a grid of values for :math:`X_j` and :math:`X_k`. The resulting 3D surface plot represents how the model's prediction changes over the joint range of these two features.

The 3D plot offers a more intuitive visualization of feature interactions compared to 2D contour plots, allowing for a better understanding of the combined effects of features on the model's predictions. The surface plot is particularly useful when you need to capture complex relationships that might not be apparent in 2D.

- **Feature Interaction Visualization:** The 3D PDP provides a comprehensive view of the interaction between two features. The resulting surface plot allows for the visualization of how the model’s output changes when the values of two features are varied simultaneously, making it easier to understand complex interactions.
- **Enhanced Interpretation:** 3D PDPs offer enhanced interpretability in scenarios where feature interactions are not linear or where the effect of one feature depends on the value of another. The 3D visualization makes these dependencies more apparent.

.. raw:: html

    <div style="height: 40px;"></div>


.. [1] Funnell, A., Shpaner, L., & Petousis, P. (2024). *Model Tuner* (Version 0.0.28b) [Software]. Zenodo. `https://doi.org/10.5281/zenodo.12727322 <https://doi.org/10.5281/zenodo.12727322>`_.
