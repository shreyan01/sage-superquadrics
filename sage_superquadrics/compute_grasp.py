"""compute_grasp(): antipodal grasp candidates from a fitted superquadric,
via numeric gradient of the same implicit function used for fitting."""
import numpy as np
from .superquadric import inside_outside, local_to_world

APPROACH_NAMES = {'x': 'side_pinch_x', 'y': 'side_pinch_y', 'z': 'vertical_pinch_z'}


def _local_surface_point(params, eta, omega):
    from .superquadric import _fexp
    e1, e2 = params['eps1'], params['eps2']
    a1, a2, a3 = params['a1'], params['a2'], params['a3']

    def sc(a, e): return _fexp(np.cos(a), e)
    def ss(a, e): return _fexp(np.sin(a), e)

    x = a1 * sc(eta, e1) * sc(omega, e2)
    y = a2 * sc(eta, e1) * ss(omega, e2)
    z = a3 * ss(eta, e1)
    return np.array([x, y, z])


def _local_normal_numeric(params, local_point, h=1e-5):
    identity_pose = dict(params, cx=0.0, cy=0.0, cz=0.0, roll=0.0, pitch=0.0, yaw=0.0)
    grad = np.zeros(3)
    for i in range(3):
        d = np.zeros(3); d[i] = h
        f_plus = inside_outside((local_point + d)[None, :], identity_pose)[0]
        f_minus = inside_outside((local_point - d)[None, :], identity_pose)[0]
        grad[i] = (f_plus - f_minus) / (2 * h)
    norm = np.linalg.norm(grad)
    return grad / norm if norm > 1e-9 else grad


def candidate_grasps(params, gripper_min_width=0.015, gripper_max_width=0.09):
    axis_angle_pairs = {
        'x': ((0.0, 0.0), (0.0, np.pi)),
        'y': ((0.0, np.pi / 2), (0.0, -np.pi / 2)),
        'z': ((np.pi / 2, 0.0), (-np.pi / 2, 0.0)),
    }
    candidates = []
    for axis, ((etaA, omA), (etaB, omB)) in axis_angle_pairs.items():
        pA_local = _local_surface_point(params, etaA, omA)
        pB_local = _local_surface_point(params, etaB, omB)
        nA_local = _local_normal_numeric(params, pA_local)
        nB_local = _local_normal_numeric(params, pB_local)
        pA_world = local_to_world(pA_local[None, :], params['cx'], params['cy'], params['cz'],
                                   params['roll'], params['pitch'], params['yaw'])[0]
        pB_world = local_to_world(pB_local[None, :], params['cx'], params['cy'], params['cz'],
                                   params['roll'], params['pitch'], params['yaw'])[0]
        nA_world = local_to_world(nA_local[None, :], 0, 0, 0,
                                   params['roll'], params['pitch'], params['yaw'])[0]
        nB_world = local_to_world(nB_local[None, :], 0, 0, 0,
                                   params['roll'], params['pitch'], params['yaw'])[0]
        width = float(np.linalg.norm(pA_world - pB_world))
        antipodal_quality = float(-np.dot(nA_world, nB_world))
        candidates.append({'axis': axis, 'approach': APPROACH_NAMES[axis],
                           'contact_a': pA_world, 'contact_b': pB_world,
                           'normal_a': nA_world, 'normal_b': nB_world,
                           'width': width, 'antipodal_quality': antipodal_quality,
                           'center': (pA_world + pB_world) / 2,
                           'feasible': gripper_min_width <= width <= gripper_max_width})
    return candidates


def rank_grasps(candidates, mode=None, history_weight=0.4, min_attempts=3):
    feasible = [c for c in candidates if c['feasible']]
    if not feasible:
        return []
    for c in feasible:
        score = c['antipodal_quality']
        if mode is not None:
            rates = mode.grasp_success_rates()
            counts = {}
            for rec in mode.grasp_records:
                counts.setdefault(rec['approach'], 0)
                counts[rec['approach']] += 1
            if c['approach'] in rates and counts.get(c['approach'], 0) >= min_attempts:
                score = (1 - history_weight) * score + history_weight * rates[c['approach']]
        c['score'] = score
    return sorted(feasible, key=lambda c: -c['score'])


def compute_grasp(params, gripper_min_width=0.015, gripper_max_width=0.09, mode=None):
    candidates = candidate_grasps(params, gripper_min_width, gripper_max_width)
    return rank_grasps(candidates, mode=mode)
