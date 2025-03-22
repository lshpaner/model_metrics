.. _changelog:   

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

\

Changelog
=========

Version 0.0.2a
----------------

* Add ``show_ks_curve`` function and enhance ``summarize_model_performance`` by @lshpaner in https://github.com/lshpaner/model_metrics/pull/1
* Add ``plot_threshold_metrics`` Function by @lshpaner in https://github.com/lshpaner/model_metrics/pull/2
* Add ``pr_feature_plot`` and Update ``roc_feature_plot`` for Enhanced Visualization by @lshpaner in https://github.com/lshpaner/model_metrics/pull/3
* Reg table enhance by @lshpaner in https://github.com/lshpaner/model_metrics/pull/4
* Rmvd (%) from MAPE header by @lshpaner in https://github.com/lshpaner/model_metrics/pull/5
* Moved roc legend to lower right default by @lshpaner in https://github.com/lshpaner/model_metrics/pull/6
* Allow Flexible Inputs and Save Behavior for ``show_roc_curve()`` by @lshpaner in https://github.com/lshpaner/model_metrics/pull/7
* Prcurve calc tests by @lshpaner in https://github.com/lshpaner/model_metrics/pull/8
* Removed unused imports and functions by @lshpaner in https://github.com/lshpaner/model_metrics/pull/9
* changed saving nomenclature in ``show_confusion_matrix`` by @lshpaner in https://github.com/lshpaner/model_metrics/pull/10
* Fix Calibration Curve Grid Plot Behavior and Update Model Nomenclature by @lshpaner in https://github.com/lshpaner/model_metrics/pull/11
* Improved support for multiple models and group categories in calibration curve by @lshpaner in https://github.com/lshpaner/model_metrics/pull/13
* Upd. ``plot_threshold_metrics`` w/ new lookup_kwgs and legend logic by @lshpaner in https://github.com/lshpaner/model_metrics/pull/14
* Rmv. unused arguments by @lshpaner in https://github.com/lshpaner/model_metrics/pull/15
* Move PDF-related Functions from ``eda_toolkit`` to ``model_metrics`` by @lshpaner in https://github.com/lshpaner/model_metrics/pull/16

**Full Changelog**: https://github.com/lshpaner/model_metrics/compare/0.0.1a...0.0.2a

Version 0.0.1a
----------------

* Updated unit tests and ``README``
* Added ``statsmodels`` to library imports 
* Added coefficients and p-values to regression summary
* Added regression capabilities to ``summarize_model_performance``
* Added lift and gains charts
* Updated versions for earlier Python compatibility 