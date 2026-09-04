"""
The public, YOLO-style API: SAGE.load(...), .predict(...), .train(...).

Every method here is a thin wrapper around the real, tested functions
already living in this package (iterative_two_part_segment,
build_graph_from_segmentation, Registry.classify_graph_ml,
Registry.confirm_graph) -- nothing new is reimplemented here, this file
only exists to give those functions a clean, discoverable, single-object
interface instead of requiring users to know the internal pipeline.
"""
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .registry import Registry
from .superquadric import is_physically_plausible
from .iterative_segment import iterative_two_part_segment
from .pipeline import build_graph_from_segmentation
from ._download import resolve_model_path


class SAGEFitError(Exception):
    """Raised when a point cloud couldn't be fit to a physically
    plausible superquadric -- distinct from a classification error,
    which never raises (a bad fit means there's nothing to classify
    yet, not that classification itself failed)."""


@dataclass
class Prediction:
    label: str
    confidence: float
    top_k: List[tuple] = field(default_factory=list)  # [(label, confidence), ...], includes the top prediction

    def __repr__(self):
        return f"Prediction(label={self.label!r}, confidence={self.confidence:.3f})"


class SAGE:
    """
    >>> model = SAGE.load('SAGE_V2')          # downloads + caches on first use
    >>> pred = model.predict(point_cloud)      # point_cloud: (N,3) numpy array
    >>> pred.label, pred.confidence
    ('mug', 0.87)

    >>> model = SAGE()                         # fresh, untrained
    >>> model.train(point_cloud, 'mug')        # online: one confirmed example
    >>> model.rebuild_classifier()             # refresh the ML classifier from confirmed examples
    >>> model.save('my_model.json')
    """

    def __init__(self, registry: Optional[Registry] = None):
        self.registry = registry if registry is not None else Registry()

    # ------------------------------------------------------------------
    # Loading / saving
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, name_or_path: str, progress: bool = True) -> "SAGE":
        """name_or_path: a known model name (e.g. 'SAGE_V2', auto-downloaded
        and cached), or a path to an existing local .json model file.

        Note: loading rebuilds the classifier from scratch from the raw
        examples stored in the file (not a serialized model object) --
        for a full pretrained model this can take several seconds to
        tens of seconds (SAGE_V2, ~35k examples, takes ~15s). Silent
        multi-second operations looking indistinguishable from a hang
        was a real, repeated source of confusion during this project's
        development -- printing here is a direct, deliberate response
        to that, not decoration."""
        path = resolve_model_path(name_or_path, progress=progress)
        if progress:
            print(f"Loading {name_or_path} and rebuilding classifier "
                  f"(this can take several seconds to tens of seconds for a full model)...")
        import time
        t0 = time.time()
        registry = Registry.load(path)
        if progress:
            n = getattr(registry, "_ml_classifier_n_examples", None)
            suffix = f" from {n} examples" if n else ""
            print(f"Done in {time.time()-t0:.1f}s{suffix}.")
        return cls(registry=registry)

    def save(self, path: str):
        self.registry.save(path)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _fit(self, point_cloud: np.ndarray, max_nfev: int = 1500, axisymmetric: bool = False):
        point_cloud = np.asarray(point_cloud)
        if point_cloud.ndim != 2 or point_cloud.shape[1] != 3:
            raise ValueError(
                f"point_cloud must be an (N,3) array of XYZ points, got shape {point_cloud.shape}"
            )
        params_a, params_b, assignment = iterative_two_part_segment(
            point_cloud, verbose=False, max_nfev=max_nfev, axisymmetric=axisymmetric
        )
        if not is_physically_plausible(params_a):
            raise SAGEFitError(
                "Fitted shape failed the physical-plausibility check (a dimension "
                "exceeded 0.5m) -- this usually means the input point cloud doesn't "
                "isolate a single real object cleanly. Check segmentation/cropping "
                "upstream of this call."
            )
        if params_b is not None and not is_physically_plausible(params_b):
            params_b, assignment = None, None
        graph = build_graph_from_segmentation(point_cloud, params_a, params_b, assignment)
        return graph

    def predict(self, point_cloud: np.ndarray, top_k: int = 3, scoring: str = "ml",
                max_nfev: int = 1500, axisymmetric: bool = False) -> Prediction:
        """scoring: 'ml' (ExtraTrees, the classifier this project's real
        accuracy results are built on), 'ensembled', or 'joint' (the
        original single-Gaussian-per-mode rule, kept for comparison/
        reproducibility, not recommended for new use)."""
        graph = self._fit(point_cloud, max_nfev=max_nfev, axisymmetric=axisymmetric)
        if scoring == "ml":
            ranked = self.registry.classify_graph_ml(graph, top_k=top_k)
        elif scoring == "ensembled":
            ranked = self.registry.classify_graph_ensembled(graph, top_k=top_k)
        else:
            ranked = self.registry.classify_graph(graph, top_k=top_k)
        if not ranked:
            raise SAGEFitError(
                "Fit succeeded but there are no learned categories to classify "
                "against yet -- this model hasn't been trained on anything. "
                "Call .train(...) first, or load a pretrained model."
            )
        label, confidence = ranked[0]
        return Prediction(label=str(label), confidence=float(confidence), top_k=ranked)

    # ------------------------------------------------------------------
    # Online training
    # ------------------------------------------------------------------

    def train(self, point_cloud: np.ndarray, label: str, max_nfev: int = 1500,
              axisymmetric: bool = False):
        """One confirmed example, added via the real online-learning path
        (Registry.confirm_graph) -- the same running-statistics update
        used throughout this project, not a separate training mode."""
        graph = self._fit(point_cloud, max_nfev=max_nfev, axisymmetric=axisymmetric)
        return self.registry.confirm_graph(graph, label, F=1)

    def rebuild_classifier(self, n_estimators: int = 500) -> bool:
        """Rebuilds the ExtraTrees classifier from whatever confirmed
        examples exist so far (via .train() calls and/or bulk-imported
        data). Returns False if there's nothing to train on yet, or if
        scikit-learn isn't installed."""
        return self.registry.rebuild_ml_classifier(n_estimators=n_estimators)

    def import_training_data(self, X: np.ndarray, y):
        """Bulk-add (feature_vector, label) pairs without re-fitting raw
        point clouds -- for when you already have exported 13D feature
        vectors (see the project's export tooling) rather than raw
        clouds. Call .rebuild_classifier() afterward to use this data."""
        self.registry.import_ml_training_data(X, y)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def categories(self) -> List[str]:
        return sorted(self.registry.graph_modes.keys())

    def describe(self, label: str) -> str:
        return self.registry.describe_graph(label)

    def __repr__(self):
        return f"SAGE(categories={self.categories()})"
