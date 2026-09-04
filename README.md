# SAGE — Superquadric-based Adaptive Geometric Explainability

A non-neural, interpretable object recognition system for robotic
manipulation. Objects are represented as superquadrics fit directly to
point clouds via nonlinear least-squares — every number the system
reasons over is a named physical quantity (a radius, a roundness
exponent, a color hue), not a learned embedding. A category vocabulary
is learned online, with no gradient descent and no neural network
weights anywhere in the pipeline.

Classification uses a non-neural tree ensemble (ExtraTrees) over these
physically interpretable features — on held-out, video-disjoint
YCB-Video data, this reaches **89.4%** top-1 accuracy from a single
observation, and **95.7%** when multiple viewpoints of the same object
are fused before classification (5-fold cross-validated, statistically
significant improvement, p=0.0072). See the project's paper for the
full accuracy characterization, including honestly-reported limitations
and negative results.

## Install

```bash
pip install sage-superquadrics
```

## Quick start

```python
from sage_superquadrics import SAGE

# Loads a pretrained model, downloading and caching it on first use
model = SAGE.load("SAGE_V1")

# point_cloud: an (N, 3) numpy array of XYZ points, isolating one object
pred = model.predict(point_cloud)
print(pred.label, pred.confidence)   # e.g. "mug", 0.87
print(pred.top_k)                     # [("mug", 0.87), ("bowl", 0.09), ...]
```

## Online learning from scratch

```python
from sage_superquadrics import SAGE

model = SAGE()  # fresh, untrained

# Each call is one real, confirmed example -- no batching required
model.train(point_cloud_1, "mug")
model.train(point_cloud_2, "mug")
model.train(point_cloud_3, "bowl")

# Build the classifier from whatever's been confirmed so far
model.rebuild_classifier()

pred = model.predict(new_point_cloud)
model.save("my_model.json")
```

## Why superquadrics instead of a learned embedding?

Every prediction traces back to physically meaningful numbers you can
inspect directly:

```python
print(model.describe("mug"))
# I know 2 structural variant(s) of "mug":
#   - graph mode a1b2c3d4 (n=14): parts=['dominant', 'secondary_0']
#       dominant: ~81x78mm footprint, 75mm tall, eps=(0.10,0.91), ...
```

When something misclassifies, you can look at *why* — a specific
fitted dimension, a roundness exponent that hit its optimizer bound —
not just a confidence score with no further explanation available.
This is the actual trade-off, stated plainly: a full raw-point-cloud
black-box baseline (PointNet) outperforms SAGE on raw accuracy under
the same evaluation protocol (93.2% vs. 89.4%). SAGE's contribution is
interpretability and a non-neural, physically-grounded representation,
not a claim of beating black-box accuracy.

## What's in this package

The core fitting, online-learning, and classification pipeline only —
`SAGE`, `Registry`, superquadric fitting, and graph-based multi-part
matching. Dataset-specific training/evaluation tooling (used to produce
the accuracy numbers above) lives in the
[GitHub repository](https://github.com/shreyan01/pnsg_superquadrics),
not in this package, to keep the installable library small and
general-purpose.

## Citation

If you use this in research, please cite the accompanying paper (see
the GitHub repository for the current reference).

## License

Apache-2.0
