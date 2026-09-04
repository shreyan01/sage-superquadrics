"""
SAGE -- Superquadric-based Adaptive Geometric Explainability.

A non-neural, interpretable object recognition system for robotics:
objects are represented as superquadrics fit directly to point clouds
via nonlinear least-squares, and a category vocabulary is learned
online (no gradient descent, no neural network weights anywhere in the
pipeline).

    >>> from sage_superquadrics import SAGE
    >>> model = SAGE.load('SAGE_V2')
    >>> pred = model.predict(point_cloud)
    >>> pred.label, pred.confidence
"""
from .model import SAGE, Prediction, SAGEFitError
from .registry import Registry, canonicalize

__version__ = "1.0.0"

__all__ = ["SAGE", "Prediction", "SAGEFitError", "Registry", "canonicalize", "__version__"]
