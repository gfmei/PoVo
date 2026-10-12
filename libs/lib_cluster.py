"""Graph grouping utilities for constrained quality spectral clustering."""

import torch
import torch.nn.functional as F


def split_disconnected_clusters(adjacency_matrix, cluster_labels):
    """Keep each instance connected and give isolated superpoints separate IDs.

    The requested spectral cluster count may be smaller than the graph's
    number of components. In that case spectral assignments can combine
    unrelated or unobserved nodes; those assignments are not instance links.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    within_cluster = cluster_labels[:, None] == cluster_labels[None, :]
    edges = ((adjacency_matrix > 0) & within_cluster).detach().cpu().numpy()
    _, labels = connected_components(csr_matrix(edges), directed=False)
    return torch.as_tensor(labels, dtype=torch.long, device=cluster_labels.device)


def compute_cluster_edge_weights_optimized(adjacency_matrix, cluster_labels, num_clusters, eps=1e-6):
    """
    Compute the average edge weights between different clusters using vectorized operations.

    Args:
        adjacency_matrix (torch.Tensor): Adjacency matrix of shape (N, N).
        cluster_labels (torch.Tensor): Cluster labels for each superpoint, shape (N,).
        num_clusters (int): Number of clusters.
        eps (float): Small epsilon value to prevent division by zero.

    Returns:
        cluster_edge_weights (torch.Tensor): Average edge weights between clusters, shape (num_clusters, num_clusters).
    """

    # One-hot encode cluster labels
    cluster_one_hot = F.one_hot(cluster_labels, num_classes=num_clusters).float()  # Shape: (N, num_clusters)

    # Compute the sum of edge weights between clusters
    cluster_edge_weight_sums = cluster_one_hot.t() @ adjacency_matrix @ cluster_one_hot  # Shape: (num_clusters, num_clusters)

    # Include absent edges in the average. A single boundary connection must
    # not count as strong agreement between every member of two clusters.
    cluster_sizes = cluster_one_hot.sum(0)
    cluster_edge_counts = cluster_sizes[:, None] * cluster_sizes[None, :]

    # Compute the average edge weight between clusters
    cluster_edge_weights = cluster_edge_weight_sums / (cluster_edge_counts + eps)

    # Ensure symmetry
    cluster_edge_weights = (cluster_edge_weights + cluster_edge_weights.t()) / 2
    cluster_edge_weights.fill_diagonal_(0)

    return cluster_edge_weights


def compute_cluster_confidence_scores(cluster_labels, element_confidence_scores):
    """
    Compute confidence scores for each cluster based on the confidence scores of its elements (superpoints or clusters).

    Args:
        cluster_labels (torch.Tensor): Labels for each element (superpoint or cluster), shape (N,).
        element_confidence_scores (torch.Tensor): Confidence scores for each element, shape (N,).

    Returns:
        cluster_confidence_scores (torch.Tensor): Confidence scores for each cluster, shape (num_clusters,).
    """

    # Get unique cluster labels and inverse indices
    unique_clusters, inverse_indices = torch.unique(cluster_labels, return_inverse=True)
    num_clusters = unique_clusters.size(0)

    # Create cluster indicator matrix
    cluster_indicator = F.one_hot(inverse_indices, num_classes=num_clusters).float()  # Shape: (N, num_clusters)

    # Compute number of elements per cluster
    num_elements_per_cluster = cluster_indicator.sum(dim=0)  # Shape: (num_clusters,)

    # Compute sum of element confidence scores per cluster
    sum_element_confidences = cluster_indicator.t() @ element_confidence_scores  # Shape: (num_clusters,)

    # Compute average confidence scores per cluster
    avg_confidences = sum_element_confidences / (num_elements_per_cluster + 1e-6)  # Shape: (num_clusters,)

    # Clamp the confidence scores to be between 0.0 and 1.0
    cluster_confidence_scores = torch.clamp(avg_confidences, min=0.0, max=1.0)

    return cluster_confidence_scores
