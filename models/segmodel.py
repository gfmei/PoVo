import copy

import torch
from torch import nn
import numpy as np

from libs.o3d_util import generate_mesh_from_pcd, get_super_point_cloud, get_spt_centers


def assign_features(p, q, q_features):
    """
    Assign features from Q to P based on the closest point distance.

    Parameters:
    - p: Point cloud P with shape (k, 3).
    - q: Point cloud Q with shape (n, 3).
    - q_features: Features associated with Q with shape (n, d).

    Returns:
    - Assigned features for P with shape (k, d).
    """
    # Calculate the pairwise squared Euclidean distance
    dist_squared = torch.cdist(p.to(q), q)
    # Find the indices of the closest points in Q for each point in P
    _, indices = torch.min(dist_squared, dim=1)
    # Assign the features from Q to P based on these indices
    assigned_features = q_features[indices]

    return assigned_features


def com_tf(lb_list):
    tf_list = [label.sum(dim=0, keepdim=True) for label in lb_list]
    return tf_list


def com_idf(lb_list):
    lb_mean = torch.stack(lb_list, dim=0).mean(dim=1)
    idf = len(lb_list) / (1 + torch.cat(lb_mean, dim=0).sum(dim=0))
    return torch.log(idf)


def get_seg_max_feat(d_xyz, xyz, feats, sptids, idx):
    spMask = np.where(sptids == idx)[0]
    seg_points = copy.deepcopy(d_xyz)
    seg_points = seg_points[spMask]
    # seg_feats = copy.deepcopy(feats)
    seg_feats = assign_features(seg_points, xyz, feats.detach().clone())
    return seg_points, seg_feats, spMask


class PoVot(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def forward(self, points, normals, scores):
        new_points = list()
        if points.dim() == 2:
            mesh = generate_mesh_from_pcd(points, normals)
            d_points, sptids = get_super_point_cloud(mesh)
            centroids, uni_ids = get_spt_centers(points, sptids)
            labels = torch.zeros((d_points.shape[0], 1), dtype=torch.long, device=points.device)
            for idx in uni_ids:
                seg_points, seg_scores, spMask = get_seg_max_feat(d_points, points, scores, sptids, idx)
                labels[spMask] = torch.argmax(seg_scores.mean(dim=0), dim=1)
                new_points.append(seg_points)
        else:
            raise NotImplementedError

        return torch.cat(new_points, dim=0), labels


def max_vote(points, normals, scores):
    new_points = list()
    if points.dim() == 2:
        mesh = generate_mesh_from_pcd(points, normals)
        d_points, sptids = get_super_point_cloud(mesh)
        centroids, uni_ids = get_spt_centers(points, sptids)
        labels = torch.zeros((d_points.shape[0], 1), dtype=torch.int, device=points.device)
        for idx in uni_ids:
            seg_points, seg_scores, spMask = get_seg_max_feat(d_points, points, scores, sptids, idx)
            labels[spMask] = torch.argmax(seg_scores.mean(dim=0).view(-1, 1), dim=0).to(torch.int)
            new_points.append(seg_points)
    else:
        raise NotImplementedError

    return torch.cat(new_points, dim=0), labels


