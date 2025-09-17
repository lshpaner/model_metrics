from .model_calculator import *
from .model_evaluator import *
from .partial_dependence import *
from .logo import *

import sys
import builtins


# Detailed Documentation


detailed_doc = """
Welcome to Model Metrics! Model Metrics is a versatile Python 
library designed to streamline the evaluation and interpretation of machine 
learning models. It provides a robust framework for generating predictions, 
computing model metrics, analyzing feature importance, and visualizing results. 
Whether you're working with SHAP values, model coefficients, confusion matrices, 
ROC curves, precision-recall plots, and other key performance indicators.

PyPI: https://pypi.org/project/model-metrics/
Documentation: https://lshpaner.github.io/model_metrics_docs/


Version: 0.0.4a5

"""

# Assign only the detailed documentation to __doc__
__doc__ = detailed_doc


__version__ = "0.0.4a5"
__author__ = "Leonid Shpaner"
__email__ = "lshpaner@ucla.edu"


# Define the custom help function
def custom_help(obj=None):
    """
    Custom help function to dynamically include ASCII art in help() output.
    """
    if (
        obj is None or obj is sys.modules[__name__]
    ):  # When `help()` is called for this module
        print(model_metrics_logo)  # Print ASCII art first
        print(detailed_doc)  # Print the detailed documentation
    else:
        original_help(obj)  # Use the original help for other objects


# Backup the original help function
original_help = builtins.help

# Override the global help function in builtins
builtins.help = custom_help
