import torch
import torch.nn.functional as F

from libs.lib_graph import gmm_params, farthest_point_sample, \
    index_points, sinkhorn_rpm


def wkmeans(x, n_clusters, dst='feats', iters=10, in_iters=5, tau=0.01):
    """
    Weighted k-means clustering using Sinkhorn for soft assignment.
    Optimized to avoid redundant computations.

    Args:
        x (torch.Tensor): Input tensor of shape (B, N, D).
        n_clusters (int): Number of clusters.
        dst (str): Distance metric ('eu' for Euclidean, 'feats' for cosine similarity).
        iters (int): Number of iterations for k-means.
        in_iters (int): Number of Sinkhorn iterations.

    Returns:
        gamma (torch.Tensor): Soft assignments of shape (B, N, num_clusters).
        pi (torch.Tensor): Cluster weights of shape (B, num_clusters).
        centroids (torch.Tensor): Cluster centroids of shape (B, num_clusters, D).
    """
    B, N, D = x.shape
    ids = farthest_point_sample(x, n_clusters, is_center=True)
    centroids = index_points(x, ids)
    gamma = torch.zeros(B, N, n_clusters, device=x.device, requires_grad=False)

    for _ in range(iters):
        if dst == 'eu':
            # Euclidean distance
            dist = torch.cdist(x, centroids)
            shift = dist.mean(dim=(-1, -2), keepdim=True)  # Shape: (B, 1, 1)
            log_gamma = (shift - dist) / tau
        else:
            # Cosine similarity
            x = F.normalize(x, p=2, dim=-1)
            centroids = F.normalize(centroids, p=2, dim=-1)
            log_gamma = torch.einsum('bnd,bmd->bnm', x, centroids)

        # Soft assignment using Sinkhorn
        gamma = sinkhorn_rpm(log_gamma, n_iters=in_iters)

        # Update cluster parameters (GMM-like)
        pi, centroids = gmm_params(gamma, x)

    return gamma, pi, centroids


def spectrum_clustering(adj_batch, n_clusters=10, k=2, dst='eu', eps=1e-5, n_iters=20, in_iters=32, tau=0.01,
                        is_auto=False):
    """
    Perform spectral clustering on a batch of adjacency matrices.

    Args:
        adj_batch (torch.Tensor): Adjacency matrices of shape (B, M, M).
        n_clusters (int): Number of clusters.
        k (int): Number of eigenvectors to extract.
        dst (str): Distance metric for k-means ('eu' or 'feats').
        eps (float): Numerical stability epsilon.
        n_iters (int): Number of iterations for k-means.
        in_iters (int): Number of Sinkhorn iterations.

    Returns:
        cluster_labels (torch.Tensor): Cluster labels of shape (B, M).
    """
    B, M, _ = adj_batch.shape

    # Degree matrix and Laplacian
    diag_batch = torch.sum(adj_batch, dim=2).diag_embed().clamp(min=eps)
    laplacian_batch = diag_batch - adj_batch

    # Symmetric normalized Laplacian
    inv_sqrt_diag = torch.diag_embed(torch.pow(torch.diagonal(diag_batch, dim1=-2, dim2=-1), -0.5))
    sym_laplacian_batch = inv_sqrt_diag.bmm(laplacian_batch).bmm(inv_sqrt_diag + eps)

    # Ensure symmetry
    sym_laplacian_batch = (sym_laplacian_batch + sym_laplacian_batch.transpose(-1, -2)) / 2
    # Regularization
    # Add regularization to the Laplacian batch
    I = torch.eye(M, device=sym_laplacian_batch.device).unsqueeze(0)  # Shape: (1, M, M)
    sym_laplacian_batch += I * 1e-3

    # Eigendecomposition
    try:
        e, v = torch.linalg.eigh(sym_laplacian_batch)  # Use symmetric decomposition
    except torch.linalg.LinAlgError:
        # Fallback to general eigendecomposition
        e, v = torch.linalg.eig(sym_laplacian_batch)
        e = e.real
        v = v.real
    if is_auto:
        sorted_e, indices = torch.sort(e, dim=-1)
        eigengap = sorted_e[:, 1:k] - sorted_e[:, :k - 1]
        optimal_k = eigengap.argmax(dim=-1) + 1  # Largest eigengap + 1

        # Perform clustering for each graph in the batch
        cluster_labels = []
        for b in range(B):
            k = optimal_k[b].item()
            selected_v = v[b, :, :k]
            norm_v = F.normalize(selected_v, p=2, dim=1)
            # Apply k-means
            gamma = wkmeans(norm_v.unsqueeze(0), n_clusters=k, dst=dst,
                            iters=n_iters, in_iters=in_iters, tau=tau)[0]
            labels = torch.argmax(gamma, dim=-1).squeeze(0)
            cluster_labels.append(labels)
        return torch.stack(cluster_labels)
    # Select the k smallest eigenvectors
    else:
        _, idx = torch.topk(e, k=k, dim=-1, largest=False)
        selected_v = torch.gather(v, 2, idx.unsqueeze(1).expand(-1, M, -1))
        # Normalize eigenvectors
        norm_v = F.normalize(selected_v, p=2, dim=1)
        # Apply k-means
        gamma = wkmeans(norm_v, n_clusters, dst=dst, iters=n_iters, in_iters=in_iters, tau=tau)[0]

        return torch.argmax(gamma, dim=-1)


def perform_spectral_clustering_on_point_cloud(adj_matrix, spt_labels, n_clusters=10, k=5, tau=0.01, is_auto=False):
    """
    Perform spectral clustering on the adjacency matrix and project superpoint labels onto the point cloud.

    Args:
        adj_matrix (torch.Tensor): Adjacency matrix of shape (num_superpoints, num_superpoints).
        spt_labels (torch.Tensor): Superpoint labels for each point, shape (num_points,).
        n_clusters (int): Number of clusters for spectral clustering.
        k (int): Number of eigenvectors for spectral clustering.
        is_auto (bool): Whether to automatically determine the number of clusters.

    Returns:
        point_labels (torch.Tensor): Cluster labels for each point in the point cloud, shape (num_points,).
        cluster_labels (torch.Tensor): Cluster labels for each superpoint, shape (num_superpoints,).
    """
    # Step 1: Perform spectral clustering on the adjacency matrix
    # print("Adjacency_matrix (Averages):\n", adj_matrix)
    # print("Min:", torch.min(adj_matrix))
    # print("Max:", torch.max(adj_matrix, dim=-1)[0].sum(), adj_matrix.shape[0])
    # print("Mean:", torch.mean(adj_matrix))
    # print("Median:", torch.median(adj_matrix))
    cluster_labels = spectrum_clustering(
        adj_matrix.unsqueeze(0), n_clusters=n_clusters, k=k, is_auto=is_auto, tau=tau
    ).squeeze(0)  # Shape: (num_superpoints,)

    # Step 2: Project cluster labels to the original point cloud
    point_labels = cluster_labels[spt_labels]

    return point_labels, cluster_labels


def threshold_cluster_edge_weights(cluster_edge_weights, threshold):
    """
    Threshold the cluster edge weights to create a binary adjacency matrix.

    Args:
        cluster_edge_weights (torch.Tensor): Edge weights between clusters, shape (C, C).
        threshold (float): Threshold for merging clusters.

    Returns:
        cluster_adjacency (torch.Tensor): Binary adjacency matrix, shape (C, C).
    """
    # Make sure we are dealing with floating type
    cluster_edge_weights = cluster_edge_weights.float()

    # Replace non-finite values with 0
    cluster_edge_weights = torch.where(
        torch.isfinite(cluster_edge_weights),
        cluster_edge_weights,
        torch.zeros_like(cluster_edge_weights)
    )

    # Apply the threshold
    cluster_adjacency = cluster_edge_weights * (cluster_edge_weights > threshold).to(cluster_edge_weights)

    return cluster_adjacency


def merge_clusters_connected_components(cluster_adjacency):
    """
    Merge clusters by finding connected components using Union-Find (Disjoint Set Union).

    Args:
        cluster_adjacency (torch.Tensor): Binary adjacency matrix, shape (C, C).

    Returns:
        merged_cluster_labels (torch.Tensor): Merged cluster labels for each cluster, shape (C,).
    """
    C = cluster_adjacency.shape[0]
    device = cluster_adjacency.device

    # Initialize parent pointers
    parent = torch.arange(C, device=device)

    # Get edges from the adjacency matrix
    row_indices, col_indices = torch.where(cluster_adjacency > 1e-3)

    # Union the connected clusters
    for u, v in zip(row_indices.tolist(), col_indices.tolist()):
        # Find roots of u and v
        pu = u
        pv = v
        # Iteratively find the root parent of u
        while parent[pu] != pu:
            parent[pu] = parent[parent[pu]]  # Path compression
            pu = parent[pu]
        # Iteratively find the root parent of v
        while parent[pv] != pv:
            parent[pv] = parent[parent[pv]]  # Path compression
            pv = parent[pv]

        if pu != pv:
            parent[pu] = pv  # Union

    # Finalize parent pointers
    for i in range(C):
        pi = i
        while parent[pi] != pi:
            parent[pi] = parent[parent[pi]]  # Path compression
            pi = parent[pi]
        parent[i] = pi

    # Relabel to ensure labels are consecutive
    _, new_labels = torch.unique(parent, return_inverse=True)
    merged_cluster_labels = new_labels

    return merged_cluster_labels


def hierarchical_agglomerative_clustering_threshold(adj_matrix, cut_threshold=0.5):
    """
    Perform hierarchical agglomerative clustering on a given similarity (adjacency) matrix
    and return cluster labels once no merges remain below the specified cut_threshold in distance.

    Args:
        adj_matrix (torch.Tensor): A (C, C) similarity matrix, where adj_matrix[i, j]
                                   is the similarity between cluster i and j.
                                   Should be symmetric and have zero diagonal.
        cut_threshold (float): Distance threshold at which to stop merging.
                               Since distance = 1 - similarity, a lower threshold
                               means we only merge very similar clusters.

    Returns:
        labels (torch.Tensor): A tensor of shape (C,) assigning each point to a cluster label.
    """
    C = adj_matrix.shape[0]
    device = adj_matrix.device

    # Convert similarity to distance
    dist_matrix = 1.0 - adj_matrix
    dist_matrix.fill_diagonal_(float('inf'))

    parent = torch.arange(C, device=device)
    cluster_size = torch.ones(C, device=device)
    active_mask = torch.ones(C, dtype=torch.bool, device=device)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    while True:
        i_indices = torch.arange(C, device=device)
        upper_triangle_mask = (i_indices.unsqueeze(1) < i_indices.unsqueeze(0)) & active_mask.unsqueeze(
            1) & active_mask.unsqueeze(0)

        # Fill all inactive or lower triangle positions with inf
        masked_dist = dist_matrix.masked_fill(~upper_triangle_mask, float('inf'))

        min_val = masked_dist.min()
        if torch.isinf(min_val):
            # No more merges possible
            break

        # If the minimum distance is greater than the threshold, stop merging
        if min_val > cut_threshold:
            break

        # Merge the closest pair
        min_pos = (masked_dist == min_val).nonzero(as_tuple=False)[0]
        ci, cj = min_pos[0].item(), min_pos[1].item()

        # Find representatives
        ci = find(ci)
        cj = find(cj)

        if ci == cj:
            # Already in the same cluster, set distance to inf and continue
            dist_matrix[ci, cj] = float('inf')
            dist_matrix[cj, ci] = float('inf')
            continue

        # Ensure ci < cj for consistency
        if ci > cj:
            ci, cj = cj, ci

        # Single-linkage update:
        dist_matrix[ci, :] = torch.min(dist_matrix[ci, :], dist_matrix[cj, :])
        dist_matrix[:, ci] = torch.min(dist_matrix[:, ci], dist_matrix[:, cj])

        # Union
        parent[cj] = ci
        cluster_size[ci] = cluster_size[ci] + cluster_size[cj]

        # Deactivate cluster cj
        active_mask[cj] = False
        dist_matrix[cj, :] = float('inf')
        dist_matrix[:, cj] = float('inf')

    # After no merges remain under threshold, find final labels
    for i in range(C):
        parent[i] = find(i)

    _, new_labels = torch.unique(parent, return_inverse=True)
    return new_labels


def map_merged_labels_to_points(superpoint_labels, cluster_labels, merged_cluster_labels):
    """
    Map the merged cluster labels to each point in the point cloud.

    Args:
        superpoint_labels (torch.Tensor): Superpoint labels for each point, shape (N,).
        cluster_labels (torch.Tensor): Initial cluster labels for each superpoint, shape (C,).
        merged_cluster_labels (torch.Tensor): Merged cluster labels for each cluster, shape (C,).

    Returns:
        point_labels (torch.Tensor): Merged cluster labels for each point, shape (N,).
    """
    # Map superpoints to merged clusters
    superpoint_merged_labels = merged_cluster_labels[cluster_labels]

    # Map points to merged cluster labels
    point_labels = superpoint_merged_labels[superpoint_labels]

    return point_labels


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

    # Compute the number of edges between clusters
    adjacency_indicator = (adjacency_matrix > 0).float()  # Indicator matrix where edges exist
    cluster_edge_counts = cluster_one_hot.t() @ adjacency_indicator @ cluster_one_hot  # Shape: (num_clusters, num_clusters)

    # Compute the average edge weight between clusters
    cluster_edge_weights = cluster_edge_weight_sums / (cluster_edge_counts + eps)

    # Ensure symmetry
    cluster_edge_weights = (cluster_edge_weights + cluster_edge_weights.t()) / 2

    return cluster_edge_weights


def compute_cluster_edge_weights_max(adjacency_matrix, cluster_labels, num_clusters):
    """
    Compute the maximum edge weights between clusters using vectorized operations.

    Args:
        adjacency_matrix (torch.Tensor): Adjacency matrix of shape (N, N).
        cluster_labels (torch.Tensor): Cluster labels for each node, shape (N,).
        num_clusters (int): Number of clusters.

    Returns:
        cluster_edge_weights (torch.Tensor): Maximum edge weights between clusters, shape (num_clusters, num_clusters).
    """
    device = adjacency_matrix.device

    # N = adjacency_matrix.shape[0]

    # Get indices and weights of non-zero edges
    edge_indices = adjacency_matrix.nonzero(as_tuple=False)  # Shape: (num_edges, 2)
    edge_weights = adjacency_matrix[edge_indices[:, 0], edge_indices[:, 1]]  # Shape: (num_edges,)

    # Get cluster labels for nodes at each end of the edges
    edge_clusters_i = cluster_labels[edge_indices[:, 0]]  # Shape: (num_edges,)
    edge_clusters_j = cluster_labels[edge_indices[:, 1]]  # Shape: (num_edges,)

    # Compute cluster pair indices
    cluster_pair_indices = edge_clusters_i * num_clusters + edge_clusters_j  # Shape: (num_edges,)

    # Initialize cluster_edge_weights_flat
    cluster_edge_weights_flat = torch.full(
        (num_clusters * num_clusters,), float(0), device=device
    )

    # Use scatter_reduce to compute max over cluster pairs
    # Available in PyTorch 1.10 and later
    cluster_edge_weights_flat.scatter_reduce_(
        0, cluster_pair_indices, edge_weights, reduce='amax'
    )

    # Reshape to (num_clusters, num_clusters)
    cluster_edge_weights = cluster_edge_weights_flat.view(num_clusters, num_clusters)

    # Ensure symmetry
    cluster_edge_weights = torch.max(cluster_edge_weights, cluster_edge_weights.t())

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
    # device = element_confidence_scores.device

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


def perform_clustering_and_merge_clusters(
        adj_matrix, spt_labels, superpoint_confidence_scores,
        n_clusters=10, k=5, thres=0.5, tau=0.01, is_auto=False
):
    """
    Perform spectral clustering on the adjacency matrix, project labels to points,
    compute edge weights between clusters, and merge clusters based on edge weights.
    Also compute and return confidence scores for each cluster.

    Args:
        adj_matrix (torch.Tensor): Adjacency matrix of shape (num_superpoints, num_superpoints).
        spt_labels (torch.Tensor): Superpoint labels for each point, shape (num_points,).
        superpoint_confidence_scores (torch.Tensor): Confidence scores for each superpoint, shape (num_superpoints,).
        n_clusters (int): Number of clusters for spectral clustering.
        k (int): Number of eigenvectors for spectral clustering.
        thres (float): Threshold for merging clusters.
        tau (float): Regularization parameter for spectral clustering.
        is_auto (bool): Whether to automatically determine the number of clusters.

    Returns:
        merged_point_labels (torch.Tensor): Merged cluster labels for each point, shape (num_points,).
        merged_cluster_labels_per_superpoint (torch.Tensor): Merged cluster labels for each superpoint, shape (num_superpoints,).
        merge_confidence_scores (torch.Tensor): Confidence scores for each merged cluster, shape (num_merged_clusters,).
    """
    # Step 1: Perform spectral clustering
    point_labels, cluster_labels = perform_spectral_clustering_on_point_cloud(
        adj_matrix, spt_labels, n_clusters, k, tau, is_auto
    )
    num_clusters = cluster_labels.max().item() + 1  # Number of clusters before merging

    # Step 2: Compute cluster edge weights using the actual number of clusters
    cluster_edge_weights = compute_cluster_edge_weights_optimized(
        adj_matrix, cluster_labels, num_clusters
    )

    # Step 3: Threshold the edge weights
    cluster_adjacency = threshold_cluster_edge_weights(cluster_edge_weights, thres)

    # Step 4: Merge clusters using connected components
    merged_cluster_labels = merge_clusters_connected_components(cluster_adjacency)
    # num_merged_clusters = merged_cluster_labels.max().item() + 1

    # Step 5: Map merged cluster labels to superpoints
    # cluster_labels: shape (num_superpoints,)
    # merged_cluster_labels: shape (num_clusters_before_merging,)
    merged_cluster_labels_per_superpoint = merged_cluster_labels[cluster_labels]  # Shape: (num_superpoints,)

    # Step 6: Map merged cluster labels to points
    merged_point_labels = merged_cluster_labels_per_superpoint[spt_labels]  # Shape: (num_points,)

    # Step 7: Compute confidence scores for merged clusters
    # Compute confidence scores for merged clusters based on superpoint confidences
    merge_confidence_scores = compute_cluster_confidence_scores(
        merged_cluster_labels_per_superpoint, superpoint_confidence_scores
    )
    # merge_confidence_scores: Shape (num_merged_clusters,)

    return point_labels, merged_point_labels, merge_confidence_scores


def merge_superpoints_based_on_adj_matrix(
        adj_matrix, spt_labels, superpoint_confidence_scores,
        thres=0.5, is_agg=False
):
    """
    Directly merge superpoints based on the adjacency matrix without clustering.
    Compute connected components from the thresholded adjacency matrix and assign merged labels.
    Also compute and return confidence scores for each merged cluster.

    Args:
        adj_matrix (torch.Tensor): Adjacency matrix of shape (num_superpoints, num_superpoints).
        spt_labels (torch.Tensor): Superpoint labels for each point, shape (num_points,).
        superpoint_confidence_scores (torch.Tensor): Confidence scores for each superpoint, shape (num_superpoints,).
        is_agg (bool): Whether to automatically determine the number of clusters.
        thres (float): Threshold for merging superpoints.

    Returns:
        merged_point_labels (torch.Tensor): Merged cluster labels for each point, shape (num_points,).
        merged_superpoint_labels (torch.Tensor): Merged cluster labels for each superpoint, shape (num_superpoints,).
        merge_confidence_scores (torch.Tensor): Confidence scores for each merged cluster, shape (num_merged_clusters,).
    """

    # Step 1: Threshold the adjacency matrix
    cluster_adjacency = threshold_cluster_edge_weights(adj_matrix, thres)

    # Step 2: Merge superpoints using connected components
    if is_agg:
        merged_superpoint_labels = hierarchical_agglomerative_clustering_threshold(cluster_adjacency, thres)
    else:
        merged_superpoint_labels = merge_clusters_connected_components(cluster_adjacency)
    # Step 3: Map merged superpoint labels to points
    merged_point_labels = merged_superpoint_labels[spt_labels]  # Shape: (num_points,)

    # Step 4: Compute confidence scores for merged clusters
    merge_confidence_scores = compute_cluster_confidence_scores(
        merged_superpoint_labels, superpoint_confidence_scores
    )
    # merge_confidence_scores: Shape (num_merged_clusters,)

    return merged_point_labels, merge_confidence_scores


if __name__ == '__main__':
    # Example inputs
    point_cloud = torch.randn(1000, 3).to("cuda")
    superpoint_labels = torch.randint(0, 10, (1000,)).to("cuda")
    masks_list = [
        torch.randint(0, 2, (50, 64, 64), dtype=torch.bool).to("cuda"),
        torch.randint(0, 2, (60, 64, 64), dtype=torch.bool).to("cuda"),
    ]  # Masks for 2 frames
    mapping_list = [
        torch.randint(0, 64, (1000, 4)).to("cuda"),
        torch.randint(0, 64, (1000, 4)).to("cuda"),
    ]  # Mapping for 2 frames

    # KNN parameter
    k = 5

    # Construct the KNN graph
    # adjacency_matrix = construct_knn_graph_multi_frame_optimized(point_cloud, superpoint_labels, masks_list,
    #                                                              mapping_list, k)
    #
    # # Display adjacency matrix
    # print(adjacency_matrix)
