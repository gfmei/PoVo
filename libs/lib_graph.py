import os
import pickle
from collections import deque
from typing import Union

import numpy as np
import pycocotools.mask
import torch
import torch.nn.functional as F
# import torch_scatter
from detectron2.structures import Instances
from numba import njit
from torch import nn



def index_points(points, idx):
    """
    Array indexing, i.e. retrieves relevant points based on indices.

    Args:
        points: input points tensor, [B, N, C].
        idx: sample index tensor, [B, S]. S can be 2 dimensional.

    Returns:
        new_points: indexed points tensor, [B, S, C].
    """
    B = points.shape[0]
    device = points.device

    # Expand the idx to be compatible with points for batched indexing
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)

    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1

    # Create batch indices for indexing
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)

    # Perform batched indexing on points using batch_indices and idx
    new_points = points[batch_indices, idx, :]

    return new_points


def farthest_point_sample(xyz, n_point, is_center=False):
    """
    Farthest Point Sampling (FPS) for point cloud data on multi-GPU.

    Args:
        xyz: input point cloud tensor, [B, N, 3].
        n_point: number of points to sample.
        is_center: if True, initializes the sampling from the centroid of the point cloud.

    Returns:
        centroids: sampled point cloud indices, [B, n_point].
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, n_point, dtype=torch.long).to(device)

    # Initialize distance as a large value for each point
    distance = torch.ones(B, N).to(device) * 1e10
    batch_indices = torch.arange(B, dtype=torch.long).to(device)

    if is_center:
        # Compute the centroid of the point cloud as the starting point
        centroid = xyz.mean(1).view(B, 1, C)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    else:
        # Randomly select the initial point if is_center is False
        farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)

    for i in range(n_point):
        # Set the current farthest point as the i-th centroid
        centroids[:, i] = farthest

        # Update the centroid
        centroid = xyz[batch_indices, farthest, :].view(B, 1, C)

        # Compute distance from the current centroid to all points
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)

        # Update distances with the minimum distance for each point
        mask = dist < distance
        distance[mask] = dist[mask]

        # Choose the next farthest point (max distance)
        farthest = torch.max(distance, dim=-1)[1]

    return centroids



def resolve_overlapping_3d_masks(pred_masks, pred_scores, device="cuda:0"):
    M, N = pred_masks.shape
    # panoptic_masks = torch.clone(pred_masks)
    scores = torch.from_numpy(pred_scores)[:, None].repeat(1, N).to(device)
    scores[~pred_masks] = 0

    panoptic_masks = torch.argmax(scores, dim=0)
    return panoptic_masks


def resolve_overlapping_masks(pred_masks, pred_scores, score_thresh=0.5, device="cuda:0"):
    M, H, W = pred_masks.shape
    pred_masks = torch.from_numpy(pred_masks).to(device)
    # panoptic_masks = torch.clone(pred_masks)
    scores = torch.from_numpy(pred_scores)[:, None, None].repeat(1, H, W).to(device)
    scores[~pred_masks] = 0
    indices = ((scores == torch.max(scores, dim=0, keepdim=True).values) & pred_masks).nonzero()
    panoptic_masks = torch.zeros((M, H, W), dtype=torch.bool, device=device)
    panoptic_masks[indices[:, 0], indices[:, 1], indices[:, 2]] = True
    panoptic_masks[scores > score_thresh] = True  # if prediction score is high enough, keep the mask anyway

    # return panoptic_masks

    return panoptic_masks.detach().cpu().numpy()


def read_detectron_instances(filepath: Union[str, os.PathLike], rle_to_mask=True) -> Instances:
    with open(filepath, "rb") as fp:
        instances = pickle.load(fp)
        if rle_to_mask:
            if instances.pred_masks_rle:
                pred_masks = np.stack([pycocotools.mask.decode(rle) for rle in instances.pred_masks_rle])
                instances.pred_masks = torch.from_numpy(pred_masks).to(torch.bool)  # (M, H, W)
            else:
                instances.pred_masks = torch.empty((0, 0, 0), dtype=torch.bool)
    return instances


def compute_projected_pts_torch(pts, cam_intr, device="cuda"):
    N = pts.shape[0]
    projected_pts = torch.zeros((N, 2), dtype=torch.int64, device=device)
    fx, fy = cam_intr[0, 0], cam_intr[1, 1]
    cx, cy = cam_intr[0, 2], cam_intr[1, 2]

    z = pts[:, 2]
    projected_pts[:, 0] = torch.round(fx * pts[:, 0] / z + cx)
    projected_pts[:, 1] = torch.round(fy * pts[:, 1] / z + cy)
    return projected_pts


@njit
def compute_projected_pts(pts, cam_intr):
    N = pts.shape[0]
    projected_pts = np.empty((N, 2), dtype=np.int64)
    fx, fy = cam_intr[0, 0], cam_intr[1, 1]
    cx, cy = cam_intr[0, 2], cam_intr[1, 2]
    for i in range(pts.shape[0]):
        z = pts[i, 2]
        x = int(np.round(fx * pts[i, 0] / z + cx))
        y = int(np.round(fy * pts[i, 1] / z + cy))
        projected_pts[i, 0] = x
        projected_pts[i, 1] = y
    return projected_pts


def compute_visibility_mask_torch(pts, projected_pts, depth_im, depth_thresh=0.005, device="cuda"):
    im_h, im_w = depth_im.shape
    visibility_mask = torch.zeros(projected_pts.shape[0], dtype=torch.bool, device=device)

    z = pts[:, 2]
    x, y = projected_pts[:, 0], projected_pts[:, 1]

    cond = (
            (x >= 0)
            & (y < im_w)
            & (y >= 0)
            & (y < im_h)
            & (depth_im[y, x] > 0)
            & (torch.abs(z - depth_im[y, x]) < depth_thresh)
    )
    visibility_mask[cond] = 1
    return visibility_mask


@njit
def compute_visibility_mask(pts, projected_pts, depth_im, depth_thresh=0.005):
    im_h, im_w = depth_im.shape
    visibility_mask = np.zeros(projected_pts.shape[0]).astype(np.bool8)
    for i in range(projected_pts.shape[0]):
        x, y = projected_pts[i]
        z = pts[i, 2]
        if x < 0 or x >= im_w or y < 0 or y >= im_h:
            continue
        if depth_im[y, x] == 0:
            continue
        if np.abs(z - depth_im[y, x]) < depth_thresh:
            visibility_mask[i] = True
    return visibility_mask


def compute_visible_masked_pts_torch(scene_pts, projected_pts, visibility_mask, pred_masks, device="cuda"):
    N = scene_pts.shape[0]
    M, _, _ = pred_masks.shape  # (M, H, W)
    masked_pts = torch.zeros((M, N), dtype=torch.bool, device=device)
    visible_indices = torch.nonzero(visibility_mask).view(-1)

    x_arr, y_arr = projected_pts[visible_indices, 0], projected_pts[visible_indices, 1]

    masked_pts[:, visible_indices] = pred_masks[:, y_arr, x_arr]
    # m_ind, y_ind, x_ind = torch.nonzero(pred_masks, as_tuple=True)
    return masked_pts
    pred_masks[y_arr, x_arr]

    for m in range(M):
        for i in visible_indices:
            x, y = projected_pts[i]
            if pred_masks[m, y, x]:
                masked_pts[m, i] = True
    return masked_pts


@njit
def compute_visible_masked_pts(scene_pts, projected_pts, visibility_mask, pred_masks):
    N = scene_pts.shape[0]
    M, _, _ = pred_masks.shape  # (M, H, W)
    masked_pts = np.zeros((M, N), dtype=np.bool_)
    visible_indices = np.nonzero(visibility_mask)[0]
    for m in range(M):
        for i in visible_indices:
            x, y = projected_pts[i]
            if pred_masks[m, y, x]:
                masked_pts[m, i] = True
    return masked_pts


# Fit 40GB
def compute_relation_matrix_self(instance_pt_count, spp, sieve):
    if not torch.is_tensor(instance_pt_count):
        instance_pt_count = torch.from_numpy(instance_pt_count)
    torch.cuda.empty_cache()

    #### Small tweak make it work on scannetpp ~ 40GB A100
    # n = instance_pt_count.shape[1]
    # numbers = list(range(n))
    # chosen_numbers = random.sample(numbers, n // max(1,int(((n *instance_pt_count.shape[0])/1e8))))
    # instance_pt_mask = instance_pt_count[:,chosen_numbers].to(torch.bool).to(torch.float16)

    # torch.cuda.empty_cache()
    # intersection = []
    # for i in range(instance_pt_mask.shape[0]):
    #     it = []
    #     for j in range(instance_pt_mask.shape[0]):
    #         it.append(instance_pt_mask[i].cuda() @ instance_pt_mask.T[:, j].cuda())
    #         torch.cuda.empty_cache()
    #     intersection.append(torch.tensor(it))  # save mem
    # intersection = torch.stack(intersection).cuda()
    # (1k,1M) ~ 1e9

    instance_pt_mask = instance_pt_count.to(torch.bool)
    instance_pt_mask_tmp = (instance_pt_mask.to(torch.float64) * sieve.expand(instance_pt_mask.shape[0], -1).to(
        torch.float64).cuda()).to(torch.float64)
    intersection = (instance_pt_mask.to(torch.float64) @ instance_pt_mask_tmp.T.to(torch.float64)).to(torch.float64)
    inliers = instance_pt_mask_tmp.sum(1, keepdims=True).to(torch.float64).cuda()
    union = (inliers + inliers.T - intersection).to(torch.float64)
    iou_matrix = intersection / (union + 1e-6)

    if (iou_matrix < 0.0).sum() > 0:
        print('wrong assert')
        breakpoint()  # wrong assert

    precision_matrix = intersection / (inliers.T + 1e-6)
    recall_matrix = intersection / (inliers + 1e-6)
    torch.cuda.empty_cache()
    return iou_matrix.to(torch.float64), precision_matrix, recall_matrix.to(torch.float64)


### Fast but Memory cost !

def compute_relation_matrix_self_mem(instance_pt_count):
    if not torch.is_tensor(instance_pt_count):
        instance_pt_count = torch.from_numpy(instance_pt_count)
    instance_pt_mask = instance_pt_count.to(torch.bool).to(torch.float32)
    intersection = instance_pt_mask @ instance_pt_mask.T  # (M, num_instances)
    inliers = instance_pt_mask.sum(1, keepdims=True)
    union = inliers + inliers.T - intersection
    iou_matrix = intersection / (union + 1e-6)
    precision_matrix = intersection / (inliers.T + 1e-6)
    recall_matrix = intersection / (inliers + 1e-6)
    return iou_matrix, precision_matrix, recall_matrix


def find_connected_components(adj_matrix):
    if torch.is_tensor(adj_matrix):
        adj_matrix = adj_matrix.detach().cpu().numpy()
    assert adj_matrix.shape[0] == adj_matrix.shape[1], "adjacency matrix should be a square matrix"

    N = adj_matrix.shape[0]
    clusters = []
    visited = np.zeros(N, dtype=np.bool_)
    for i in range(N):
        if visited[i]:
            continue
        cluster = []
        queue = deque([i])
        visited[i] = True
        while queue:
            j = queue.popleft()
            cluster.append(j)
            for k in np.nonzero(adj_matrix[j])[0]:
                if not visited[k]:
                    queue.append(k)
                    visited[k] = True
        clusters.append(cluster)
    return clusters


def sinkhorn_rpm(log_alpha, n_iters: int = 5, slack: bool = True, eps: float = -1) -> torch.Tensor:
    """ Run sinkhorn iterations to generate a near doubly stochastic matrix, where each row or column sum to <=1

    Args:
        log_alpha: log of positive matrix to apply sinkhorn normalization (B, J, K)
        n_iters (int): Number of normalization iterations
        slack (bool): Whether to include slack row and column
        eps: eps for early termination (Used only for handcrafted RPM). Set to negative to disable.

    Returns:
        log(perm_matrix): Doubly stochastic matrix (B, J, K)

    Modified from original source taken from:
        Learning Latent Permutations with Gumbel-Sinkhorn Networks
        https://github.com/HeddaCohenIndelman/Learning-Gumbel-Sinkhorn-Permutations-w-Pytorch
    """

    # Sinkhorn iterations
    prev_alpha = None
    if slack:
        zero_pad = nn.ZeroPad2d((0, 1, 0, 1))
        log_alpha_padded = zero_pad(log_alpha[:, None, :, :])

        log_alpha_padded = torch.squeeze(log_alpha_padded, dim=1)

        for i in range(n_iters):
            # Row normalization
            log_alpha_padded = torch.cat((
                log_alpha_padded[:, :-1, :] - (torch.logsumexp(log_alpha_padded[:, :-1, :], dim=2, keepdim=True)),
                log_alpha_padded[:, -1, None, :]),  # Don't normalize last row
                dim=1)
            log_alpha_padded = torch.nan_to_num(log_alpha_padded, nan=0.0)
            # Column normalization
            log_alpha_padded = torch.cat((
                log_alpha_padded[:, :, :-1] - (torch.logsumexp(log_alpha_padded[:, :, :-1], dim=1, keepdim=True)),
                log_alpha_padded[:, :, -1, None]),  # Don't normalize last column
                dim=2)
            log_alpha_padded = torch.nan_to_num(log_alpha_padded, nan=0.0)
            if eps > 0:
                if prev_alpha is not None:
                    abs_dev = torch.abs(torch.exp(log_alpha_padded[:, :-1, :-1]) - prev_alpha)
                    if torch.max(torch.sum(abs_dev, dim=[1, 2])) < eps:
                        break
                prev_alpha = torch.exp(log_alpha_padded[:, :-1, :-1]).clone()

        log_alpha = log_alpha_padded[:, :-1, :-1]
    else:
        for i in range(n_iters):
            # Row normalization (i.e. each row sum to 1)
            log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=2, keepdim=True))
            log_alpha = torch.nan_to_num(log_alpha, nan=0.0)
            # Column normalization (i.e. each column sum to 1)
            log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=1, keepdim=True))
            log_alpha = torch.nan_to_num(log_alpha, nan=0.0)

            if eps > 0:
                if prev_alpha is not None:
                    abs_dev = torch.abs(torch.exp(log_alpha) - prev_alpha)
                    if torch.max(torch.sum(abs_dev, dim=[1, 2])) < eps:
                        break
                prev_alpha = torch.exp(log_alpha).clone()
    _, max_indices = torch.max(log_alpha, dim=-1)
    return F.one_hot(max_indices, num_classes=log_alpha.size(-1)).to(log_alpha)


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
    mu = gamma.transpose(1, 2) @ pts / npi.unsqueeze(2)
    if return_sigma:
        # diff: B feats N feats J feats D
        diff = pts.unsqueeze(2) - mu.unsqueeze(1)
        # sigma: B feats J feats 3 feats 3
        eye = torch.eye(D).unsqueeze(0).unsqueeze(1).to(gamma.device)
        sigma = (((diff.unsqueeze(3) @ diff.unsqueeze(4)).squeeze() *
                  gamma).sum(dim=1) / npi).unsqueeze(2).unsqueeze(3) * eye
        return pi, mu, sigma
    return pi, mu


def log_boltzmann_kernel(log_alpha, u, v, epsilon):
    kernel = (log_alpha + u.unsqueeze(-1) + v.unsqueeze(-2)) / epsilon
    return kernel


def sinkhorn(log_alpha, p=None, q=None, epsilon=1e-2, thresh=1e-2, n_iters=10):
    # Initialise approximation vectors in log domain
    if p is None or q is None:
        batch_size, num_x, num_y = log_alpha.shape
        device = log_alpha.device
        if p is None:
            p = torch.empty(batch_size, num_x, dtype=torch.float,
                            requires_grad=False, device=device).fill_(1.0 / num_x).squeeze()
        if q is None:
            q = torch.empty(batch_size, num_y, dtype=torch.float,
                            requires_grad=False, device=device).fill_(1.0 / num_y).squeeze()
    u = torch.zeros_like(p).to(p)
    v = torch.zeros_like(q).to(q)
    # Stopping criterion, sinkhorn iterations
    for i in range(n_iters):
        u0, v0 = u, v
        # u^{l+1} = a / (K v^l)
        K = log_boltzmann_kernel(log_alpha, u, v, epsilon)
        u_ = torch.log(p + 1e-8) - torch.logsumexp(K, dim=-1)
        u = epsilon * u_ + u
        # v^{l+1} = b / (K^T u^(l+1))
        Kt = log_boltzmann_kernel(log_alpha, u, v, epsilon).transpose(-2, -1)
        v_ = torch.log(q + 1e-8) - torch.logsumexp(Kt, dim=-1)
        v = epsilon * v_ + v
        # Size of the change we have performed on u
        diff = torch.sum(torch.abs(u - u0), dim=-1) + torch.sum(torch.abs(v - v0), dim=-1)
        mean_diff = torch.mean(diff)
        if mean_diff.item() < thresh:
            break
    # Transport plan pi = diag(a)*K*diag(b)
    K = log_boltzmann_kernel(log_alpha, u, v, epsilon)
    gamma = torch.exp(K)
    return gamma


def morton_indices_from_coordinates(coords, num_bits):
    """
    Convert 3D coordinates to Morton (Z-order) curve indices.

    Args:
        coords (torch.Tensor): Tensor of shape (N, 3) with integer coordinates in [0, 2^num_bits - 1].
        num_bits (int): Number of bits to represent each coordinate.

    Returns:
        morton_indices (torch.Tensor): Tensor of shape (N,) with Morton curve indices.
    """
    coords = coords.long()
    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]

    morton_indices = interleave_bits(x, y, z, num_bits)
    return morton_indices

def interleave_bits(x, y, z, num_bits):
    """
    Interleave bits of x, y, and z to generate Morton codes.

    Args:
        x, y, z (torch.Tensor): Coordinate components, each of shape (N,).
        num_bits (int): Number of bits.

    Returns:
        codes (torch.Tensor): Morton codes, shape (N,).
    """
    codes = torch.zeros_like(x, dtype=torch.long)
    for i in range(num_bits):
        bit_mask = 1 << i
        xi = (x & bit_mask) << (2 * i)
        yi = (y & bit_mask) << (2 * i + 1)
        zi = (z & bit_mask) << (2 * i + 2)
        codes |= xi | yi | zi
    return codes