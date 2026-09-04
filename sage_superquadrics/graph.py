from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from .superquadric import fit_superquadric

@dataclass
class PartNode:
    node_id: str
    params: dict
    role: Optional[str] = None
    color_features: Optional[tuple] = None   # (hue_degrees, saturation) or None
    taper_features: Optional[tuple] = None   # (r_bottom, r_top) or None -- see radius_profile.compute_taper_features
    @property
    def centroid(self):
        return np.array([self.params['cx'], self.params['cy'], self.params['cz']])
    @property
    def principal_axis(self):
        from .superquadric import local_to_world
        tip = local_to_world(np.array([[0, 0, 1.0]]), 0, 0, 0, self.params['roll'], self.params['pitch'], self.params['yaw'])[0]
        return tip / (np.linalg.norm(tip) + 1e-9)
    def size_along_axis(self): return self.params['a3']
    def horizontal_extent(self): return max(self.params['a1'], self.params['a2'])

def classify_relation(a, b):
    offset = b.centroid - a.centroid
    dist = float(np.linalg.norm(offset)); horiz = float(np.linalg.norm(offset[:2])); vert = float(offset[2])
    axis_dot = float(np.dot(a.principal_axis, b.principal_axis)); coaxial = abs(axis_dot) > 0.9
    if horiz > 0.4 * a.horizontal_extent() and abs(vert) < a.size_along_axis(): attach = 'side'
    elif vert > 0.6 * a.size_along_axis(): attach = 'top'
    elif vert < -0.6 * a.size_along_axis(): attach = 'bottom'
    else: attach = 'overlapping_or_unclear'
    return {'attachment': attach, 'coaxial': coaxial, 'distance': dist, 'offset': offset.tolist()}

@dataclass
class ObjectGraph:
    nodes: list = field(default_factory=list)
    edges: list = field(default_factory=list)
    def add_node(self, node): self.nodes.append(node)
    def build_edges_mst(self):
        self.edges = []
        if len(self.nodes) < 2: return
        in_tree = {self.nodes[0].node_id}; remaining = {n.node_id: n for n in self.nodes[1:]}
        id_to_node = {n.node_id: n for n in self.nodes}
        while remaining:
            best = None
            for in_id in in_tree:
                for out_id, out_node in remaining.items():
                    d = np.linalg.norm(id_to_node[in_id].centroid - out_node.centroid)
                    if best is None or d < best[0]: best = (d, in_id, out_id)
            _, a_id, b_id = best
            relation = classify_relation(id_to_node[a_id], id_to_node[b_id])
            self.edges.append((a_id, b_id, relation)); in_tree.add(b_id); del remaining[b_id]
    def describe(self):
        lines = [f'Object structure: {len(self.nodes)} part(s), {len(self.edges)} relation(s)']
        for n in self.nodes:
            role = n.role or '(unlabeled)'
            lines.append(f'  node {n.node_id} [{role}]: size=({n.params["a1"]*1000:.0f},{n.params["a2"]*1000:.0f},{n.params["a3"]*1000:.0f})mm eps=({n.params["eps1"]:.2f},{n.params["eps2"]:.2f})')
        for a_id, b_id, rel in self.edges:
            a_role = next(n.role or a_id for n in self.nodes if n.node_id == a_id)
            b_role = next(n.role or b_id for n in self.nodes if n.node_id == b_id)
            coax = ', coaxial' if rel['coaxial'] else ''
            lines.append(f'  {a_role} -> {b_role}: {rel["attachment"]} ({rel["distance"]*1000:.0f}mm away{coax})')
        return '\n'.join(lines)

def fit_graph_from_segmented_clouds(labeled_clouds):
    graph = ObjectGraph()
    for i, (role, cloud) in enumerate(labeled_clouds.items()):
        fitted, info = fit_superquadric(cloud)
        graph.add_node(PartNode(node_id=f'n{i}', params=fitted, role=role))
    graph.build_edges_mst()
    return graph