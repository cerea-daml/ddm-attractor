from .linear_downstream import LinearRegressionModule
from .mlp_regression import MLPRegressionModule
from .sgd_regression import SGDRegressionModule

__all__ = [
    "LinearRegressionModule",
    "MLPRegressionModule",
    "SGDRegressionModule"
]
