"""Deterministic initialization helpers for four-start Lloyd rounding."""

import torch


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
