import numpy as np
from .iterative_segment import iterative_two_part_segment
from .graph import PartNode, ObjectGraph
from .compute_grasp import compute_grasp

def build_graph_from_segmentation(raw_cloud, params_a, params_b, assignment, color_features=None, taper_features=None):
    """color_features and taper_features both apply to the DOMINANT
    part only -- a deliberate scoping decision, not an oversight.
    Tracking these per-PART would meaningfully increase complexity for
    most real objects; the dominant part is a reasonable proxy for the
    whole object's signature."""
    graph = ObjectGraph()
    graph.add_node(PartNode(node_id='n0', params=params_a, role='dominant',
                             color_features=color_features, taper_features=taper_features))
    if params_b is not None:
        graph.add_node(PartNode(node_id='n1', params=params_b, role='secondary_0'))
    graph.build_edges_mst()
    return graph

def process_scene(raw_cloud, spoken_noun, mu_det, loop, feedback_fn,
                   gripper_min_width=0.015, gripper_max_width=0.09, verbose=True, max_iters=8, max_nfev=3000):
    params_a, params_b, assignment = iterative_two_part_segment(raw_cloud, verbose=False, max_iters=max_iters, max_nfev=max_nfev)
    graph = build_graph_from_segmentation(raw_cloud, params_a, params_b, assignment)
    record = loop.step_graph(graph, spoken_noun, mu_det, feedback_fn)
    result = {'graph': graph, 'gate_record': record, 'grasp': None}
    if not record['triggered']:
        mode = None
        if record.get('matched_mode'):
            for gm in loop.registry.graph_modes.get(spoken_noun, []):
                if gm.mode_id == record['matched_mode']:
                    mode = gm.part_modes.get('dominant'); break
        ranked = compute_grasp(params_a, gripper_min_width, gripper_max_width, mode=mode)
        result['grasp'] = ranked
    return result