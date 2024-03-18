import copy

import numpy as np
import torch

from libs.lib_utils import farthest_point_sample, index_points, sinkhorn, angle_difference


def pairwise_histogram_distance_pytorch(hist1, hist2):
    """
    Calculate the pairwise histogram intersection distance between batches of point clouds A and B,
    optimized for PyTorch with support for batch processing.

    Parameters:
    - histograms_A: Histograms for each point in point clouds A, shape: [B, N, D].
    - histograms_B: Histograms for each point in point clouds B, shape: [B, M, D].

    Returns:
    - distance_matrix: A tensor of distances where element (b, i, j) is the distance from the i-th point in the b-th point cloud in A to the j-th point in the b-th point cloud in B.
    """
    # Ensure input tensors are float for division
    hist1 = hist1.float()
    hist2 = hist2.float()

    # Expand histograms for broadcasting: [B, N, 1, D] and [B, 1, M, D]
    hist1_exp = hist1.unsqueeze(2)
    hist2_exp = hist2.unsqueeze(1)

    # Calculate minimum of each pair of histograms, resulting in a [B, N, M, D] tensor
    minima = torch.min(hist1_exp, hist2_exp)

    # Sum over the last dimension (D) to get the intersection values, resulting in a [B, N, M] tensor
    intersections = torch.sum(minima, dim=-1)

    # Normalize the intersections to get a similarity measure and then convert to distances
    sum1 = torch.sum(hist1, dim=-1, keepdim=True)  # Shape [B, N, 1]
    sum2 = torch.sum(hist2, dim=-1, keepdim=True)  # Shape [B, 1, M]
    max_sum = torch.min(sum1, sum2.transpose(1, 2))  # Shape [B, N, M]

    normalized_intersections = intersections / max_sum
    distance_matrix = 1 - normalized_intersections

    return distance_matrix


def pairwise_histogram_distance_optimized(hist1, hist2):
    """
    Calculate the pairwise histogram intersection distance between each point in A to every point in B,
    optimized to reduce explicit for loops.

    Parameters:
    - histograms_A: Histograms for each point in point cloud A (shape: [N, D]).
    - histograms_B: Histograms for each point in point cloud B (shape: [M, D]).

    Returns:
    - distance_matrix: A matrix of distances where element (i, j) is the distance from the i-th point in A to the j-th point in B.
    """
    # Expand histograms_A and histograms_B to 3D tensors for broadcasting
    # histograms_A: [N, 1, D], histograms_B: [1, M, D]
    hist1_exp = hist1[:, np.newaxis, :]
    hist2_exp = hist2[np.newaxis, :, :]

    # Calculate minimum of each pair of histograms (using broadcasting), resulting in a [N, M, D] tensor
    minima = np.minimum(hist1_exp, hist2_exp)

    # Sum over the last dimension (D) to get the intersection values, resulting in a [N, M] matrix
    intersections = np.sum(minima, axis=2)

    # Calculate normalized intersections as distances
    sum1 = np.sum(hist1, axis=1)[:, np.newaxis]  # Shape [N, 1]
    sum2 = np.sum(hist2, axis=1)[np.newaxis, :]  # Shape [1, M]
    max_sum = np.minimum(sum1, sum2)  # Broadcasting to get max sum for each pair

    normalized_intersections = intersections / max_sum
    distance_matrix = 1 - normalized_intersections

    return distance_matrix


def gmm_params(gamma, pts, return_sigma=False):
    """
    gamma: B feats N feats J
    pts: B feats N feats D
    """
    # pi: B feats J
    D = pts.size(-1)
    pi = gamma.mean(dim=1)
    npi = pi * gamma.shape[1] + 1e-5
    # p: B feats J feats D
    mu = gamma.transpose(1, 2) @ pts / npi.float().unsqueeze(2)
    if return_sigma:
        # diff: B feats N feats J feats D
        diff = pts.unsqueeze(2) - mu.unsqueeze(1)
        # sigma: B feats J feats 3 feats 3
        eye = torch.eye(D).unsqueeze(0).unsqueeze(1).to(gamma.device)
        sigma = (((diff.unsqueeze(3) @ diff.unsqueeze(4)).squeeze() *
                  gamma).sum(dim=1) / npi).unsqueeze(2).unsqueeze(3) * eye
        return pi, mu, sigma
    return pi, mu


def clu_dis(pts, mu, d_type='eu'):
    if d_type == 'ang':
        return angle_difference(pts, mu)
    elif d_type == 'hist':
        return pairwise_histogram_distance_pytorch(pts, mu)
    else:
        dis = torch.cdist(pts, mu)
        # min_d = torch.min(dis, dim=-1)[0].min(dim=-1)[0]
        # max_d = torch.max(dis, dim=-1)[0].max(dim=-1)[0]
        # gap = (max_d - min_d).clip(min=1e-4)
        # dis = (dis - min_d.view(-1, 1, 1)) / gap.view(-1, 1, 1)
        return dis


def fusion_wkeans(feat_list, w_list, n_clus=20, iters=30, is_prob=False, idx=0):
    bs, num, dim = feat_list[idx].shape
    device = feat_list[0].device
    ids = farthest_point_sample(feat_list[idx], n_clus)
    cts_list = [index_points(copy.deepcopy(feat), ids) for feat in feat_list]
    gamma, pi = torch.ones((bs, num, n_clus), device=device), None
    mask_th = np.sqrt(3) * torch.topk(
        torch.cdist(cts_list[idx], cts_list[idx]), k=2, largest=False, dim=-1)[0][:, :, 1].mean(dim=-1)
    if w_list[idx] is None:
        w_list[idx] = 1.0 / mask_th.clip(min=1e-4).expand(bs, num, n_clus)

    for _ in range(iters):
        cost_list = list()
        mask_dist = None
        cost = None
        for i in range(len(feat_list)):
            dist_i = w_list[i] * clu_dis(feat_list[i], cts_list[i])
            if i == idx:
                mask_dist = dist_i
            cost_list.append(dist_i)
            cost = torch.stack(cost_list, dim=0)
        if len(feat_list) > 1:
            cost = cost.sum(dim=0)
        else:
            cost = cost[0]
        if is_prob:
            gamma = sinkhorn(cost, q=pi, max_iter=10)[0]
        else:
            gamma = sinkhorn(cost, q=None, max_iter=10)[0]
        mask = (mask_dist < mask_th).to(gamma)
        gamma = mask * gamma
        gamma = gamma / gamma.sum(dim=-1, keepdim=True).clip(min=1e-3)
        cts_list = [gmm_params(gamma, feat)[1] for feat in feat_list]
        pi = gamma.mean(dim=1)

    return gamma, pi, cts_list


if __name__ == '__main__':
    pts = torch.rand(3, 1024, 32)
    fusion_wkeans([pts, pts], [2.0, 1.0], 20)
