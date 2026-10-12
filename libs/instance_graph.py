"""Morton/Hilbert point neighbors combined with visible frame-mask evidence."""

import torch

from libs.instance_clustering import frame_mask_evidence, modulate_geometry
from libs.graph_affinity import graph_appearance_settings, histogram_rgb_affinity
from libs.spatial_neighbors import DEFAULT_CURVE_ORDERS, superpoint_curve_adjacency


def construct_superpoint_adjacency_graph_radius(
        pcds, spt_labels, masks_list, mapping_list, sam_scores_list,
        min_points=5, radius_threshold=.2, point_feas=None, mask_thres=.3,
        sim_thres=.8, curve_window_size=32, curve_resolution=.01,
        curve_chunk_size=8192, curve_orders=DEFAULT_CURVE_ORDERS,
        return_evidence=False, minimum_conflict_fraction=0., *, appearance_mode='clip',
        class_features=None, point_rgb=None, mask_features_list=None, logit_scale=100., mask_top_m=5,
        semantic_top_k=5, semantic_weight=.5, visual_weight=.25, rgb_weight=.25, rgb_sigma=.2):
    """Require frame agreement for each verified boundary-distance edge.

    Rows follow sorted unique superpoint IDs. Filtered or unobserved nodes
    remain isolated, with zero confidence; cannot-links always remove edges.
    """
    if not (len(masks_list) == len(mapping_list) == len(sam_scores_list)):
        raise ValueError('each frame requires masks, a mapping, and confidence scores')
    spt_labels = spt_labels.to(pcds.device).long()
    unique, inverse, counts = torch.unique(spt_labels, return_inverse=True, return_counts=True)
    valid = counts >= min_points
    geometry = torch.zeros((len(unique), len(unique)), device=pcds.device)
    if valid.any():
        valid_points = valid[inverse]
        adjacency = superpoint_curve_adjacency(
            pcds[valid_points], spt_labels[valid_points], radius=radius_threshold,
            window_size=curve_window_size, resolution=curve_resolution,
            chunk_size=curve_chunk_size, curve_orders=curve_orders)
        indices = torch.where(valid)[0]
        geometry[indices[:, None], indices[None, :]] = adjacency
    evidence = frame_mask_evidence(
        spt_labels, masks_list, mapping_list, sam_scores_list, geometry,
        mask_threshold=mask_thres, minimum_conflict_fraction=minimum_conflict_fraction)
    appearance = graph_appearance_settings(dict(
        appearance_mode=appearance_mode, semantic_top_k=semantic_top_k, mask_top_m=mask_top_m,
        semantic_weight=semantic_weight, visual_weight=visual_weight,
        rgb_weight=rgb_weight, rgb_sigma=rgb_sigma))
    if appearance_mode == 'histogram_rgb':
        appearance.pop('appearance_mode')
        adjacency = histogram_rgb_affinity(
            geometry, evidence.agreement, spt_labels, point_features=point_feas,
            class_features=class_features, point_rgb=point_rgb,
            masks_list=masks_list, mappings=mapping_list, scores_list=sam_scores_list,
            mask_features_list=mask_features_list,
            logit_scale=logit_scale, similarity_threshold=sim_thres,
            chunk_size=curve_chunk_size, **appearance)
    elif point_feas is None:
        adjacency = geometry * evidence.agreement
    else:
        adjacency = modulate_geometry(geometry, evidence.agreement, point_feas,
                                     spt_labels, similarity_threshold=sim_thres)
    adjacency = adjacency.masked_fill(evidence.cannot_link, 0.)
    confidence = evidence.confidence.masked_fill(~valid, 0.)
    result = adjacency, confidence
    return (*result, evidence) if return_evidence else result
