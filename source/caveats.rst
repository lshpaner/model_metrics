
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

Caveats
============

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
- If ``score``is not passed, the function will look up the first item in ``model.scoring`` (if available).
- If neither a custom threshold nor a valid model threshold is available, the default value of ``0.5`` is used.

