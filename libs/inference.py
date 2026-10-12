"""Shared frame selection, feature fusion, and instance scoring."""

import argparse

import numpy as np
import cv2
import torch
import torch.nn.functional as F


def resize_depth_and_intrinsic(depth, intrinsic, image_dim):
    """Keep native depth calibration consistent when resizing pixel coordinates."""
    source_height, source_width = depth.shape
    width, height = image_dim
    if (source_width, source_height) == (width, height):
        return depth, intrinsic
    if intrinsic is not None:
        intrinsic = np.asarray(intrinsic, dtype=np.float32).copy()
        sx, sy = width / source_width, height / source_height
        intrinsic[0, 0] *= sx
        intrinsic[1, 1] *= sy
        intrinsic[0, 2] = (intrinsic[0, 2] + .5) * sx - .5
        intrinsic[1, 2] = (intrinsic[1, 2] + .5) * sy - .5
    return cv2.resize(depth, (width, height), interpolation=cv2.INTER_NEAREST), intrinsic


def parse_frame_count(value):
    """Parse a positive view budget, or 'all' as zero (no view limit)."""
    if value == 'all':
        return 0
    try:
        count = int(value)
    except (TypeError, ValueError):
        count = 0
    if count < 1:
        raise argparse.ArgumentTypeError('frames must be a positive integer or all')
    return count


def select_camera_frames(paths, reader, count):
    """Cover camera position and direction; zero selects all finite-pose views."""
    if count < 0:
        raise ValueError('frame count must be nonnegative; zero selects all views')
    if not paths or (count > 0 and len(paths) <= count):
        return paths
    poses = np.stack([reader.get_pose(path) for path in paths])
    valid = np.flatnonzero(np.isfinite(poses).all(axis=(1, 2)))
    if not len(valid):
        return []
    if count == 0:
        return [paths[int(index)] for index in valid]
    forward = poses[valid, :3, 2].copy()
    forward /= np.maximum(np.linalg.norm(forward, axis=1, keepdims=True), 1e-9)
    features = np.concatenate((poses[valid, :3, 3] / .5, forward * 2), axis=1)
    chosen = [0]
    distance = ((features - features[0]) ** 2).sum(1)
    while len(chosen) < min(count, len(valid)):
        distance[chosen] = -1
        index = int(distance.argmax())
        chosen.append(index)
        distance = np.minimum(distance, ((features - features[index]) ** 2).sum(1))
    return [paths[int(valid[index])] for index in sorted(chosen)]


def accumulate_mask_features(features, counter, masks, embeddings, scores, mapping,
                             chunk_size=4096):
    """Fuse only projected pixels, with one normalized contribution per view.

    Confidence fusion averages overlapping embeddings, so duplicate proposals
    and scenes with many masks cannot gain extra weight across views.
    """
    visible = torch.where(mapping[:, 3] == 1)[0]
    embeddings = torch.as_tensor(embeddings, device=features.device).float()
    if embeddings.shape != (len(masks), features.shape[1]):
        raise ValueError('cached mask embeddings do not match mask count or CLIP dimension')
    embeddings = F.normalize(torch.nan_to_num(embeddings, nan=0., posinf=0., neginf=0.), dim=-1)
    scores = torch.nan_to_num(torch.as_tensor(scores, device=features.device).float(),
                             nan=0., posinf=0., neginf=0.).clamp(0, 1)
    weights = scores.clone()
    for indices in visible.split(chunk_size):
        membership = masks[:, mapping[indices, 1], mapping[indices, 2]].T.float()
        contribution = F.normalize((membership * weights) @ embeddings, dim=-1)
        confidence = (membership * scores).amax(1, keepdim=True)
        confidence *= contribution.norm(dim=-1, keepdim=True) > 0
        features[indices] += contribution * confidence
        counter[indices] += confidence


def score_instances(point_features, text_features, masks, objectness, adapter=None, batch_size=256,
                    return_class_ids=False):
    """Use CLIP temperature and preserve geometric/mask objectness in scores."""
    if text_features is None:
        classes = torch.full((len(masks),), -1, dtype=torch.long, device=point_features.device)
        return (objectness, None, classes) if return_class_ids else (objectness, None)
    if batch_size < 1:
        raise ValueError('batch size must be positive')
    text_features = F.normalize(text_features.float(), dim=-1)
    scale = getattr(adapter, 'logit_scale', None)
    scale = scale.detach().float().exp().clamp(max=100) if scale is not None else 100.
    known = point_features.norm(dim=-1) > 0
    point_scores = torch.zeros((len(point_features), len(text_features)), device=point_features.device)
    for start in range(0, len(point_features), batch_size):
        embedding = F.normalize(point_features[start:start + batch_size].float(), dim=-1)
        point_scores[start:start + batch_size] = (scale * (embedding @ text_features.T)).softmax(-1)
    point_scores *= known[:, None]
    membership = masks.float() * known[None, :]
    instance_scores = membership @ point_scores / membership.sum(1, keepdim=True).clamp_min(1)
    semantic = point_scores.argmax(-1).masked_fill(~known, -100)
    scores, classes = instance_scores.max(1)
    classes = classes.masked_fill(membership.sum(1) == 0, -1)
    result = scores * objectness, semantic
    return (*result, classes) if return_class_ids else result
