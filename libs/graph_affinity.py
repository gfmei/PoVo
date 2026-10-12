"""Superpoint appearance affinity from predicted label histograms and RGB."""

import math
from collections.abc import Mapping

import torch
import torch.nn.functional as F


APPEARANCE_DEFAULTS = {
    'appearance_mode': 'clip',
    'semantic_top_k': 5,
    'mask_top_m': 5,
    'semantic_weight': .5,
    'visual_weight': .25,
    'rgb_weight': .25,
    'rgb_sigma': .2,
}


def graph_appearance_settings(config=None):
    """Validate settings shared by both inference paths and cache fingerprints."""
    source = config if isinstance(config, Mapping) else vars(config) if config is not None else {}
    settings = {key: source.get(key, value) for key, value in APPEARANCE_DEFAULTS.items()}
    if settings['appearance_mode'] not in ('clip', 'histogram_rgb'):
        raise ValueError('graph.appearance_mode must be clip or histogram_rgb')
    for key in ('semantic_top_k', 'mask_top_m'):
        value = settings[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f'graph.{key} must be a positive integer')
    for key in ('semantic_weight', 'visual_weight', 'rgb_weight', 'rgb_sigma'):
        value = settings[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f'graph.{key} must be a finite number')
        settings[key] = float(value)
    if min(settings[key] for key in ('semantic_weight', 'visual_weight', 'rgb_weight')) < 0:
        raise ValueError('graph appearance weights must be nonnegative')
    if sum(settings[key] for key in ('semantic_weight', 'visual_weight', 'rgb_weight')) <= 0:
        raise ValueError('at least one graph appearance weight must be positive')
    if settings['rgb_sigma'] <= 0:
        raise ValueError('graph.rgb_sigma must be positive')
    return settings


def select_superpoint_masks(labels, masks_list, mappings, *, top_m=5,
                            minimum_visibility=.1, chunk_size=4096):
    """Keep the global top-m (view, mask) observations by projected-pixel IoU.

    Rasterize each superpoint's visible pixels without counting duplicate
    projections. Rank all masks across all views; ties retain input order.
    Unobserved nodes and non-overlapping masks never enter the selection.
    """
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 1
           for value in (top_m, chunk_size)):
        raise ValueError('top_m and chunk_size must be positive integers')
    if not 0 <= minimum_visibility <= 1 or not math.isfinite(minimum_visibility):
        raise ValueError('minimum_visibility must lie between zero and one')
    if len(masks_list) != len(mappings):
        raise ValueError('each frame requires masks and a mapping')
    _, inverse, counts = torch.unique(labels, return_inverse=True, return_counts=True)
    device, count = labels.device, len(counts)
    frames = torch.full((count, top_m), -1, device=device, dtype=torch.long)
    indices = torch.full_like(frames, -1)
    overlaps = torch.zeros((count, top_m), device=device)
    for frame_id, (masks, mapping) in enumerate(zip(masks_list, mappings)):
        masks = torch.as_tensor(masks, device=device).bool()
        mapping = torch.as_tensor(mapping, device=device)
        if masks.ndim != 3 or mapping.shape != (len(labels), 4):
            raise ValueError('masks must be Q x H x W and mappings must be points x 4')
        if not len(masks):
            continue
        height, width = masks.shape[-2:]
        if height < 1 or width < 1:
            raise ValueError('masks must have positive image dimensions')
        valid = ((mapping[:, 3] == 1) & (mapping[:, 1] >= 0) & (mapping[:, 1] < height)
                 & (mapping[:, 2] >= 0) & (mapping[:, 2] < width))
        visible = torch.where(valid)[0]
        if not len(visible):
            continue
        observed = torch.bincount(inverse[visible], minlength=count) >= (
            minimum_visibility * counts).clamp_min(1)
        pixels = mapping[visible, 1].long() * width + mapping[visible, 2].long()
        unique_pixels = torch.unique(inverse[visible] * (height * width) + pixels)
        nodes = unique_pixels // (height * width)
        pixels = unique_pixels % (height * width)
        areas = torch.bincount(nodes, minlength=count)
        flat = masks.flatten(1)
        intersection = torch.zeros((count, len(masks)), device=device, dtype=torch.long)
        for start in range(0, len(pixels), chunk_size):
            membership = flat[:, pixels[start:start + chunk_size]].T.long()
            intersection.index_add_(0, nodes[start:start + chunk_size], membership)
        union = areas[:, None] + flat.sum(1)[None, :] - intersection
        iou = intersection.float() / union.clamp_min(1)
        iou *= observed[:, None]
        candidates = torch.cat((overlaps, iou), dim=1)
        candidate_frames = torch.cat((frames, torch.full_like(intersection, frame_id)), dim=1)
        candidate_masks = torch.cat((indices, torch.arange(len(masks), device=device)[None, :].expand(count, -1)), dim=1)
        order = candidates.argsort(dim=1, descending=True, stable=True)[:, :top_m]
        overlaps = candidates.gather(1, order)
        frames = candidate_frames.gather(1, order).masked_fill(overlaps <= 0, -1)
        indices = candidate_masks.gather(1, order).masked_fill(overlaps <= 0, -1)
    return frames, indices, overlaps


def superpoint_label_histograms(labels, masks_list, mappings, mask_features_list,
                               class_features, scores_list, *, top_m=5, top_k=5,
                               logit_scale=100., chunk_size=4096):
    """Build semantic histograms only from each superpoint's top-m view masks.

    Rank masks by geometric overlap before considering labels or confidence.
    Each selected mask casts top-k CLIP label-probability votes, weighted by
    its IoU and mask confidence. No fused point features supply label votes.
    """
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError('top_k must be a positive integer')
    if not math.isfinite(logit_scale) or logit_scale <= 0:
        raise ValueError('logit_scale must be finite and positive')
    if not (len(masks_list) == len(mappings) == len(mask_features_list) == len(scores_list)):
        raise ValueError('each frame requires masks, mappings, mask features, and confidence scores')
    if class_features.ndim != 2:
        raise ValueError('class features must be a matrix of text embeddings')
    device = labels.device
    classes = class_features.to(device).float()
    histograms = torch.zeros((len(torch.unique(labels)), len(classes)), device=device)
    valid_classes = torch.isfinite(classes).all(1) & (classes.norm(dim=1) > 0)
    if not len(classes) or not valid_classes.any():
        return histograms
    classes = F.normalize(torch.where(valid_classes[:, None], classes, 0.), dim=-1)
    frames, indices, overlaps = select_superpoint_masks(
        labels, masks_list, mappings, top_m=top_m, chunk_size=chunk_size)
    k = min(top_k, int(valid_classes.sum()))
    for frame_id, (features, scores, masks) in enumerate(zip(mask_features_list, scores_list, masks_list)):
        features = torch.as_tensor(features, device=device).float()
        scores = torch.as_tensor(scores, device=device).float()
        if features.shape != (len(masks), class_features.shape[1]) or scores.shape != (len(masks),):
            raise ValueError('mask features and scores must align with masks and the text encoder')
        rows, slots = torch.where(frames == frame_id)
        if not len(rows):
            continue
        mask_indices = indices[rows, slots]
        selected, reverse = torch.unique(mask_indices, return_inverse=True)
        features = features[selected]
        known = torch.isfinite(features).all(1) & (features.norm(dim=1) > 0)
        features = F.normalize(torch.where(known[:, None], features, 0.), dim=-1)
        logits = (logit_scale * (features @ classes.T)).masked_fill(~valid_classes[None, :], -float('inf'))
        categories = logits.topk(k, dim=-1).indices
        probabilities = logits.softmax(-1).gather(1, categories)
        confidence = torch.nan_to_num(scores[mask_indices], nan=0., posinf=0., neginf=0.).clamp(0, 1)
        mass = overlaps[rows, slots] * confidence * known[reverse]
        votes = probabilities[reverse] * mass[:, None]
        bins = rows[:, None] * len(classes) + categories[reverse]
        histograms.view(-1).scatter_add_(0, bins.reshape(-1), votes.reshape(-1))
    mass = histograms.sum(1, keepdim=True)
    return histograms / torch.where(mass > 0, mass, 1.)


def _pool_features(features, labels, *, require_nonzero=False):
    _, inverse, counts = torch.unique(labels, return_inverse=True, return_counts=True)
    if features.ndim != 2 or len(features) != len(labels):
        raise ValueError('appearance features must have one vector per point')
    features = features.to(labels.device).float()
    valid = torch.isfinite(features).all(1)
    if require_nonzero:
        valid &= features.norm(dim=1) > 0
    pooled = torch.zeros((len(counts), features.shape[1]), device=labels.device)
    pooled.index_add_(0, inverse, torch.where(valid[:, None], features, 0.))
    mass = torch.zeros(len(counts), device=labels.device)
    mass.index_add_(0, inverse, valid.float())
    return pooled / mass[:, None].clamp_min(1), mass > 0


def histogram_rgb_affinity(geometry, agreement, labels, *, point_features=None,
                          class_features=None, point_rgb=None, masks_list=None, mappings=None,
                          mask_features_list=None, scores_list=None, mask_top_m=5,
                          semantic_top_k=5, semantic_weight=.5, visual_weight=.25,
                          rgb_weight=.25, rgb_sigma=.2, logit_scale=100.,
                          similarity_threshold=.8, chunk_size=4096):
    """Modulate supported geometry edges by available semantic/visual/RGB cues.

    Histogram and visual similarity use cosine; RGB uses a Gaussian on mean
    colors in [0, 1]. Missing cues are omitted and weights renormalized for
    each pair. Pairs with no usable appearance evidence retain mask agreement.
    """
    graph_appearance_settings(dict(appearance_mode='histogram_rgb', semantic_top_k=semantic_top_k,
                                  mask_top_m=mask_top_m,
                                  semantic_weight=semantic_weight, visual_weight=visual_weight,
                                  rgb_weight=rgb_weight, rgb_sigma=rgb_sigma))
    if not math.isfinite(similarity_threshold) or not 0 <= similarity_threshold <= 1:
        raise ValueError('histogram/RGB similarity threshold must lie between zero and one')
    labels = labels.to(geometry.device)
    rows, columns = torch.where(torch.triu((geometry > 0) & (agreement > 0), diagonal=1))
    numerator = torch.zeros(len(rows), device=geometry.device)
    denominator = torch.zeros_like(numerator)

    def add_cue(vectors, supported, weight, *, rgb=False):
        paired = supported[rows] & supported[columns]
        if rgb:
            distances = (vectors[rows] - vectors[columns]).square().sum(-1)
            similarity = torch.exp(-distances / (2 * rgb_sigma ** 2))
        else:
            normalized = F.normalize(vectors, dim=-1)
            similarity = (normalized[rows] * normalized[columns]).sum(-1)
        numerator.add_(weight * similarity.clamp(0, 1) * paired)
        denominator.add_(weight * paired)

    if semantic_weight > 0 and class_features is not None and mask_features_list is not None:
        if masks_list is None or mappings is None or scores_list is None:
            raise ValueError('semantic mask votes require masks, mappings, and confidence scores')
        histograms = superpoint_label_histograms(
            labels, masks_list, mappings, mask_features_list, class_features, scores_list,
            top_m=mask_top_m, top_k=semantic_top_k, logit_scale=logit_scale, chunk_size=chunk_size)
        add_cue(histograms, histograms.norm(dim=1) > 0, semantic_weight)
    if point_features is not None:
        point_features = point_features.to(geometry.device)
        if visual_weight > 0:
            visual, supported = _pool_features(point_features, labels, require_nonzero=True)
            normalized = F.normalize(visual, dim=-1)
            supported &= visual.norm(dim=1) > 0
            paired = supported[rows] & supported[columns]
            similarity = ((normalized[rows] * normalized[columns]).sum(-1) + 1) / 2
            numerator.add_(visual_weight * similarity.clamp(0, 1) * paired)
            denominator.add_(visual_weight * paired)
    if rgb_weight > 0 and point_rgb is not None:
        point_rgb = torch.as_tensor(point_rgb, device=geometry.device).float()
        if point_rgb.ndim != 2 or point_rgb.shape != (len(labels), 3):
            raise ValueError('point_rgb must have shape (number of points, 3)')
        finite = point_rgb[torch.isfinite(point_rgb).all(1)]
        if finite.numel() and ((finite < 0).any() or (finite > 1).any()):
            raise ValueError('point_rgb colors must lie between zero and one')
        rgb, supported = _pool_features(point_rgb, labels)
        add_cue(rgb, supported, rgb_weight, rgb=True)
    similarity = torch.where(denominator > 0, numerator / torch.where(denominator > 0, denominator, 1.), 1.)
    similarity = torch.where(similarity >= similarity_threshold, similarity, 0.)
    weights = geometry[rows, columns] * agreement[rows, columns] * similarity
    result = torch.zeros_like(geometry)
    result[rows, columns] = result[columns, rows] = weights
    return result
