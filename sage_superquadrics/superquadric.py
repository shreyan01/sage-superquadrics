import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares

def _fexp(base, exponent):
    return np.sign(base) * (np.abs(base) ** exponent)

def world_to_local(points, cx, cy, cz, roll, pitch, yaw):
    rot = R.from_euler('zyx', [yaw, pitch, roll]).as_matrix()
    centered = points - np.array([cx, cy, cz])
    return centered @ rot

def local_to_world(points, cx, cy, cz, roll, pitch, yaw):
    rot = R.from_euler('zyx', [yaw, pitch, roll]).as_matrix()
    return points @ rot.T + np.array([cx, cy, cz])

def inside_outside(points, params):
    x, y, z = world_to_local(points, params['cx'], params['cy'], params['cz'],
                              params['roll'], params['pitch'], params['yaw']).T
    a1, a2, a3 = params['a1'], params['a2'], params['a3']
    e1, e2 = params['eps1'], params['eps2']
    tx = np.abs(x / a1) ** (2.0 / e2)
    ty = np.abs(y / a2) ** (2.0 / e2)
    inner = (tx + ty) ** (e2 / e1)
    tz = np.abs(z / a3) ** (2.0 / e1)
    return inner + tz

def radial_residual(points, params):
    F = inside_outside(points, params)
    e1 = params['eps1']
    vol_scale = np.sqrt(params['a1'] * params['a2'] * params['a3'])
    return (F ** (e1 / 2.0) - 1.0) * vol_scale

PARAM_ORDER = ['a1', 'a2', 'a3', 'eps1', 'eps2', 'cx', 'cy', 'cz', 'roll', 'pitch', 'yaw']

def params_to_vec(params):
    return np.array([params[k] for k in PARAM_ORDER])

def vec_to_params(vec):
    return dict(zip(PARAM_ORDER, vec))

def initial_guess(points):
    centroid = points.mean(axis=0)
    extents = (points.max(axis=0) - points.min(axis=0)) / 2.0
    extents = np.clip(extents, 1e-3, None)
    return {'a1': extents[0], 'a2': extents[1], 'a3': extents[2], 'eps1': 1.0, 'eps2': 1.0,
            'cx': centroid[0], 'cy': centroid[1], 'cz': centroid[2],
            'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0}

BOUNDS_LOW = {'a1': 1e-3, 'a2': 1e-3, 'a3': 1e-3, 'eps1': 0.1, 'eps2': 0.1,
              'cx': -np.inf, 'cy': -np.inf, 'cz': -np.inf, 'roll': -np.pi, 'pitch': -np.pi, 'yaw': -np.pi}
BOUNDS_HIGH = {'a1': np.inf, 'a2': np.inf, 'a3': np.inf, 'eps1': 1.9, 'eps2': 1.9,
               'cx': np.inf, 'cy': np.inf, 'cz': np.inf, 'roll': np.pi, 'pitch': np.pi, 'yaw': np.pi}

def fit_superquadric(points, init=None, verbose=0, max_size_multiplier=None,
                      min_size_multiplier=None, position_margin_multiplier=None, max_nfev=3000,
                      axisymmetric=False):
    """axisymmetric=True forces a1==a2 AND eps2==1.0 throughout the fit --
    a direct physical prior for genuinely round object classes (bottle,
    can, mug body), NOT a workaround needing multiple views. A single
    camera view only ever sees part of a round object's circumference;
    with no constraint, the optimizer 'reasonably' fits an ellipse to
    that partial arc (found repeatedly tonight: e.g. a real bottle fit
    as 120x41mm instead of round). IMPORTANT, found only after real
    data testing: a1==a2 alone is NOT sufficient for true circularity --
    a superquadric's cross-section is only a real circle when eps2 is
    ALSO near 1.0. Real trained bottle prototypes showed eps2=0.62 even
    with a1==a2 correctly forced, meaning the actual fitted cross-section
    was a rounded square ("squircle"), not a circle. Since eps2==1.0 is
    just as much a known physical fact for these categories as a1==a2,
    it is fixed the same way -- removed as a free parameter entirely,
    not left for the optimizer to discover from partial-view data."""
    if init is None:
        init = initial_guess(points)

    if axisymmetric:
        # reduced parameterization: a2 tied to a1, eps2 FIXED at 1.0 (both
        # removed as independent free parameters -- true circularity is a
        # known fact for these categories, not something to infer)
        order = ['a1', 'a3', 'eps1', 'cx', 'cy', 'cz', 'roll', 'pitch', 'yaw']
        x0 = np.array([init[k] for k in order])
        lo = np.array([BOUNDS_LOW[k] for k in order])
        hi = np.array([BOUNDS_HIGH[k] for k in order])

        def vec_to_params_axisym(vec):
            p = dict(zip(order, vec))
            p['a2'] = p['a1']
            p['eps2'] = 1.0
            return p

        if max_size_multiplier is not None or min_size_multiplier is not None:
            extents = (points.max(axis=0) - points.min(axis=0)) / 2.0
            extents = np.clip(extents, 1e-3, None)
            radial_extent = (extents[0] + extents[1]) / 2.0  # average of x/y extents for the shared radius
            for k, ext in [('a1', radial_extent), ('a3', extents[2])]:
                idx = order.index(k)
                if max_size_multiplier is not None:
                    hi[idx] = min(hi[idx], max_size_multiplier * ext)
                if min_size_multiplier is not None:
                    lo[idx] = max(lo[idx], min_size_multiplier * ext)
                x0[idx] = np.clip(x0[idx], lo[idx], hi[idx])

        if position_margin_multiplier is not None:
            centroid = points.mean(axis=0)
            box_extents = (points.max(axis=0) - points.min(axis=0)) / 2.0
            box_extents = np.clip(box_extents, 1e-3, None)
            margin = position_margin_multiplier * box_extents
            for i, k in enumerate(['cx', 'cy', 'cz']):
                idx = order.index(k)
                lo[idx] = centroid[i] - margin[i]
                hi[idx] = centroid[i] + margin[i]
                x0[idx] = np.clip(x0[idx], lo[idx], hi[idx])

        def resid_fn(vec):
            return radial_residual(points, vec_to_params_axisym(vec))

        result = least_squares(resid_fn, x0, bounds=(lo, hi), method='trf', loss='soft_l1',
                                f_scale=0.05, max_nfev=max_nfev, verbose=verbose)
        fitted = vec_to_params_axisym(result.x)
        final_resid = resid_fn(result.x)
        info = {'success': result.success, 'rmse': float(np.sqrt(np.mean(final_resid**2))), 'nfev': result.nfev}
        return fitted, info

    x0 = params_to_vec(init)
    lo = np.array([BOUNDS_LOW[k] for k in PARAM_ORDER])
    hi = np.array([BOUNDS_HIGH[k] for k in PARAM_ORDER])
    if max_size_multiplier is not None or min_size_multiplier is not None:
        extents = (points.max(axis=0) - points.min(axis=0)) / 2.0
        extents = np.clip(extents, 1e-3, None)
        for i, k in enumerate(['a1', 'a2', 'a3']):
            idx = PARAM_ORDER.index(k)
            if max_size_multiplier is not None:
                hi[idx] = min(hi[idx], max_size_multiplier * extents[i])
            if min_size_multiplier is not None:
                lo[idx] = max(lo[idx], min_size_multiplier * extents[i])
            x0[idx] = np.clip(x0[idx], lo[idx], hi[idx])
    if position_margin_multiplier is not None:
        centroid = points.mean(axis=0)
        box_extents = (points.max(axis=0) - points.min(axis=0)) / 2.0
        box_extents = np.clip(box_extents, 1e-3, None)
        margin = position_margin_multiplier * box_extents
        for i, k in enumerate(['cx', 'cy', 'cz']):
            idx = PARAM_ORDER.index(k)
            lo[idx] = centroid[i] - margin[i]
            hi[idx] = centroid[i] + margin[i]
            x0[idx] = np.clip(x0[idx], lo[idx], hi[idx])
    def resid_fn(vec):
        return radial_residual(points, vec_to_params(vec))
    result = least_squares(resid_fn, x0, bounds=(lo, hi), method='trf', loss='soft_l1',
                            f_scale=0.05, max_nfev=max_nfev, verbose=verbose)
    fitted = vec_to_params(result.x)
    final_resid = resid_fn(result.x)
    info = {'success': result.success, 'rmse': float(np.sqrt(np.mean(final_resid ** 2))), 'nfev': result.nfev}
    return fitted, info

def is_physically_plausible(params, max_dim_m=0.5):
    return (params['a1'] <= max_dim_m and params['a2'] <= max_dim_m and params['a3'] <= max_dim_m)

def sample_superquadric_surface(params, n_points=2000, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    eta = rng.uniform(-np.pi / 2, np.pi / 2, n_points)
    omega = rng.uniform(-np.pi, np.pi, n_points)
    e1, e2 = params['eps1'], params['eps2']
    a1, a2, a3 = params['a1'], params['a2'], params['a3']
    def sc(a, e): return _fexp(np.cos(a), e)
    def ss(a, e): return _fexp(np.sin(a), e)
    x = a1 * sc(eta, e1) * sc(omega, e2)
    y = a2 * sc(eta, e1) * ss(omega, e2)
    z = a3 * ss(eta, e1)
    local_pts = np.stack([x, y, z], axis=1)
    return local_to_world(local_pts, params['cx'], params['cy'], params['cz'],
                           params['roll'], params['pitch'], params['yaw'])

def crop_partial_view(points, camera_dir=np.array([0, 0, 1]), keep_fraction=0.5):
    centroid = points.mean(axis=0)
    rel = points - centroid
    rel_norm = rel / (np.linalg.norm(rel, axis=1, keepdims=True) + 1e-9)
    dot = rel_norm @ (camera_dir / np.linalg.norm(camera_dir))
    thresh = np.quantile(dot, 1.0 - keep_fraction)
    return points[dot >= thresh]

def add_noise(points, sigma=0.003, rng=None):
    if rng is None:
        rng = np.random.default_rng(1)
    return points + rng.normal(0, sigma, points.shape)