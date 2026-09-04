"""
Radius-profile-based body/neck detection for axisymmetric (round)
objects. Built after finding that the generic residual-clustering
segmentation (designed for spatially separate attached parts like a
mug's handle) doesn't work for radius tapering: a single primitive's
compromise fit to a body+neck cloud is roughly uniformly bad across
MOST of the object (62% of points exceeded the leftover threshold in a
real test), not locally bad in one small region -- so the generic
"small leftover cluster = second part" assumption breaks down.

This instead measures radius directly as a function of height along
the object's own axis (valid specifically BECAUSE the object is known
to be round: every visible point at a given height is a real
measurement of that height's true radius, regardless of which side of
the object it came from -- no occlusion ambiguity for this specific
measurement, unlike the object's overall shape).
"""
import numpy as np
from .superquadric import world_to_local, fit_superquadric


def compute_radius_profile(raw_cloud, pose_params, n_bins=15):
    """Returns (bin_centers, median_radii, bin_counts) using a pose
    estimate's center/orientation (radius itself can be a bad
    compromise value at this point -- only cx,cy,cz,roll,pitch,yaw are
    used here, not a1/a3)."""
    local = world_to_local(raw_cloud, pose_params['cx'], pose_params['cy'], pose_params['cz'],
                            pose_params['roll'], pose_params['pitch'], pose_params['yaw'])
    z = local[:, 2]
    radius = np.sqrt(local[:, 0] ** 2 + local[:, 1] ** 2)
    bins = np.linspace(z.min(), z.max(), n_bins + 1)
    centers, med_radii, counts = [], [], []
    for i in range(n_bins):
        mask = (z >= bins[i]) & (z < bins[i + 1])
        if mask.sum() < 5:
            continue
        centers.append((bins[i] + bins[i + 1]) / 2)
        med_radii.append(np.median(radius[mask]))
        counts.append(int(mask.sum()))
    return np.array(centers), np.array(med_radii), np.array(counts)


def find_radius_step(bin_centers, med_radii, min_ratio=1.8, min_abs_diff_m=0.008):
    """Finds the height at which radius drops most sharply. Returns
    (split_height, ratio) or (None, None) if no step exceeds BOTH
    min_ratio AND min_abs_diff_m. Ratio alone can be fooled by tiny,
    physically meaningless segments (a 2mm vs 4mm 'step' is technically
    a 2x ratio, but it's noise, not a neck) -- min_abs_diff_m=8mm
    requires a real, physically meaningful radius change, not just a
    proportionally large one."""
    if len(med_radii) < 4:
        return None, None
    best_idx, best_ratio = None, 1.0
    for i in range(1, len(med_radii) - 1):
        side_a_mean = med_radii[:i + 1].mean()
        side_b_mean = med_radii[i + 1:].mean()
        ratio = max(side_a_mean, side_b_mean) / max(min(side_a_mean, side_b_mean), 1e-6)
        abs_diff = abs(side_a_mean - side_b_mean)
        if ratio > best_ratio and abs_diff >= min_abs_diff_m:
            best_ratio = ratio
            best_idx = i
    if best_idx is None or best_ratio < min_ratio:
        return None, None
    split_height = (bin_centers[best_idx] + bin_centers[best_idx + 1]) / 2
    return split_height, best_ratio


def fit_body_and_neck(raw_cloud, max_nfev=1500, min_ratio=1.8, min_segment_points=50):
    """Attempts to fit an axisymmetric object as TWO stacked round
    segments (body + neck) using a direct radius-profile measurement,
    instead of generic residual clustering. Returns (params_body,
    params_neck, split_info) with params_neck=None if no genuine step
    was found. min_ratio raised from 1.3 to 1.8 after finding that 1.3
    was too permissive on REAL sensor data: a synthetic (noise-free)
    test correctly rejected a plain cylinder, but real aggregated can
    clouds triggered false-positive necks on EVERY training example
    with wildly inconsistent sizes (7-111mm), actively hurting can's
    accuracy (36.9%->28.0%) instead of leaving it untouched. Real
    depth noise apparently produces enough radius jitter to cross a
    1.3x threshold even on genuinely constant-radius objects."""
    pose_estimate, _ = fit_superquadric(raw_cloud, max_nfev=max_nfev, axisymmetric=True)
    bin_centers, med_radii, counts = compute_radius_profile(raw_cloud, pose_estimate)
    split_height, ratio = find_radius_step(bin_centers, med_radii, min_ratio=min_ratio)

    if split_height is None:
        return pose_estimate, None, {'has_neck': False}

    local = world_to_local(raw_cloud, pose_estimate['cx'], pose_estimate['cy'], pose_estimate['cz'],
                            pose_estimate['roll'], pose_estimate['pitch'], pose_estimate['yaw'])
    z = local[:, 2]
    below_mask = z <= split_height
    above_mask = ~below_mask

    if below_mask.sum() < min_segment_points or above_mask.sum() < min_segment_points:
        return pose_estimate, None, {'has_neck': False, 'reason': 'segment too small'}

    below_cloud = raw_cloud[below_mask]
    above_cloud = raw_cloud[above_mask]
    below_fit, _ = fit_superquadric(below_cloud, max_nfev=max_nfev, axisymmetric=True,
                                     max_size_multiplier=4.0, min_size_multiplier=0.05)
    above_fit, _ = fit_superquadric(above_cloud, max_nfev=max_nfev, axisymmetric=True,
                                     max_size_multiplier=4.0, min_size_multiplier=0.05)

    # 'body' = whichever segment is wider (by convention, matches how a
    # real bottle's body is the dominant/larger part)
    if below_fit['a1'] >= above_fit['a1']:
        body_fit, neck_fit = below_fit, above_fit
    else:
        body_fit, neck_fit = above_fit, below_fit

    return body_fit, neck_fit, {'has_neck': True, 'ratio': ratio, 'split_height': split_height}


def compute_taper_features(raw_cloud, pose_params):
    """Returns (r_bottom_third, r_top_third) as raw radii (meters) at
    the bottom and top thirds of the object's height, ALWAYS computed
    -- no discrete 'is there a neck' decision. Deliberate redesign
    after fit_body_and_neck()'s discrete threshold was found to
    regress real accuracy (can: 36.9%->28.0%->12.8% across two
    threshold attempts): a hard yes/no branch made every individual
    fit unstable under real single-frame sensor noise. Two continuous
    numbers, always measured, let the registry's own noise-robust
    machinery (mean+variance per category) learn what's normal -- a
    plain cylinder naturally has r_bottom~=r_top with tight learned
    variance; a real bottle naturally has r_top<<r_bottom -- without
    any fragile per-fit branching."""
    local = world_to_local(raw_cloud, pose_params['cx'], pose_params['cy'], pose_params['cz'],
                            pose_params['roll'], pose_params['pitch'], pose_params['yaw'])
    z = local[:, 2]
    radius = np.sqrt(local[:, 0] ** 2 + local[:, 1] ** 2)
    z_min, z_max = z.min(), z.max()
    z_range = max(z_max - z_min, 1e-6)
    bottom_mask = z < (z_min + z_range / 3.0)
    top_mask = z > (z_max - z_range / 3.0)
    r_bottom = float(np.median(radius[bottom_mask])) if bottom_mask.sum() >= 5 else float(np.median(radius))
    r_top = float(np.median(radius[top_mask])) if top_mask.sum() >= 5 else float(np.median(radius))
    return r_bottom, r_top


def compute_radial_profile(raw_cloud, pose_params, heights=(0.1, 0.3, 0.5, 0.7, 0.9), band_width=0.15):
    """Generalizes compute_taper_features from 2 sample points to N,
    measuring radius at several relative heights along the object's
    axis -- a real 'generalized cylinder' / surface-of-revolution
    profile, a classical, legitimate, non-neural geometric
    representation. A single superquadric can only represent one
    smooth primitive family; this instead traces the ACTUAL radius
    curve, capturing gradual shoulders (e.g. a cola bottle) that a
    sharp 2-point step measurement can miss, while a plain cylinder
    (can) still comes out as a flat, constant profile -- same
    underlying, already-proven-stable measurement mechanism, just
    sampled more finely."""
    local = world_to_local(raw_cloud, pose_params['cx'], pose_params['cy'], pose_params['cz'],
                            pose_params['roll'], pose_params['pitch'], pose_params['yaw'])
    z = local[:, 2]
    radius = np.sqrt(local[:, 0] ** 2 + local[:, 1] ** 2)
    z_min, z_max = z.min(), z.max()
    z_range = max(z_max - z_min, 1e-6)
    overall_median = float(np.median(radius))

    profile = []
    for h in heights:
        center = z_min + h * z_range
        half_band = band_width * z_range / 2.0
        mask = (z >= center - half_band) & (z <= center + half_band)
        if mask.sum() >= 5:
            profile.append(float(np.median(radius[mask])))
        else:
            profile.append(overall_median)
    return tuple(profile)