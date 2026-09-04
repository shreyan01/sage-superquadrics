import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from .superquadric import fit_superquadric, inside_outside

def per_point_residual(points, params):
    F = inside_outside(points, params)
    e1 = params['eps1']
    return np.abs(F ** (e1 / 2.0) - 1.0)

def cluster_points(points, radius, min_cluster_size):
    if len(points) == 0: return []
    tree = cKDTree(points)
    pairs = tree.query_pairs(r=radius, output_type='ndarray')
    n = len(points)
    if len(pairs) == 0:
        return [np.array([i]) for i in range(n) if min_cluster_size <= 1]
    row, col = pairs[:, 0], pairs[:, 1]
    data = np.ones(len(row))
    adj = csr_matrix((data, (row, col)), shape=(n, n))
    n_components, labels = connected_components(adj, directed=False)
    clusters = []
    for c in range(n_components):
        idx = np.where(labels == c)[0]
        if len(idx) >= min_cluster_size: clusters.append(idx)
    return clusters

def segment_and_fit(raw_cloud, residual_threshold=0.35, cluster_radius=0.01,
                     min_cluster_size=40, verbose=True, max_nfev=3000, axisymmetric=False):
    """axisymmetric=True fits BOTH the dominant part and any secondary
    parts with a1==a2, eps2==1.0 (true circularity, not just equal
    radii -- see superquadric.py). Lets a genuinely round object with a
    real neck taper (e.g. a bottle) naturally split into a body segment
    and a narrower neck segment, both correctly circular, rather than
    forcing the whole object into one constant-radius primitive that
    can't represent tapering at all."""
    dominant_params, dominant_info = fit_superquadric(raw_cloud, max_nfev=max_nfev, axisymmetric=axisymmetric)
    residuals = per_point_residual(raw_cloud, dominant_params)
    explained_mask = residuals <= residual_threshold
    leftover_idx = np.where(~explained_mask)[0]
    parts = [{'params': dominant_params, 'point_indices': np.where(explained_mask)[0],
              'role': 'dominant', 'info': dominant_info}]
    if len(leftover_idx) < min_cluster_size: return parts
    leftover_points = raw_cloud[leftover_idx]
    clusters = cluster_points(leftover_points, cluster_radius, min_cluster_size)
    for i, cluster_local_idx in enumerate(clusters):
        cluster_global_idx = leftover_idx[cluster_local_idx]
        cluster_cloud = raw_cloud[cluster_global_idx]
        if len(cluster_cloud) < 8: continue
        part_params, part_info = fit_superquadric(cluster_cloud, max_nfev=max_nfev, axisymmetric=axisymmetric)
        parts.append({'params': part_params, 'point_indices': cluster_global_idx,
                      'role': f'secondary_{i}', 'info': part_info})
    return parts