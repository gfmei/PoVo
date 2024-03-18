import nltk
import numpy as np
import torch
from nltk.stem import WordNetLemmatizer

# Ensure you have the necessary NLTK data
nltk.download('wordnet')
nltk.download('omw-1.4')
# Initialize the WordNet lemmatizer
lemmatizer = WordNetLemmatizer()


def remove_repeat_words(text):
    # Convert words to their singular form using lemmatization
    singular_words = set(lemmatizer.lemmatize(word) for word in text)

    return singular_words


def angle_difference(src_feats, dst_feats):
    """Calculate angle between each pair of vectors.
    Assumes points are l2-normalized to unit length.

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src_feats.shape
    _, M, _ = dst_feats.shape
    dist = torch.matmul(src_feats, dst_feats.permute(0, 2, 1))
    dist = torch.acos(dist)

    return dist


def index_points(points, idx):
    """Array indexing, i.e. retrieves relevant points based on indices

    Args:
        points: input points data_loader, [B, N, C]
        idx: sample index data_loader, [B, S]. S can be 2 dimensional
    Returns:
        new_points:, indexed points data_loader, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, n_point, is_center=False):
    """
    Input:
        pts: point cloud data, [B, N, 3]
        n_point: number of samples
    Return:
        sub_xyz: sampled point cloud index, [B, n_point]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, n_point, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(xyz) * 1e10
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    if is_center:
        centroid = xyz.mean(1).view(B, 1, C)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    else:
        farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    for i in range(n_point):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, C)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]

    return centroids


def angle(v1: torch.Tensor, v2: torch.Tensor):
    """Compute angle between 2 vectors

    For robustness, we use the same formulation as in PPFNet, i.e.
        angle(v1, v2) = atan2(cross(v1, v2), dot(v1, v2)).
    This handles the case where one of the vectors is 0.0, since torch.atan2(0.0, 0.0)=0.0

    Args:
        v1: (B, *, 3)
        v2: (B, *, 3)

    Returns:

    """

    cross_prod = torch.stack([v1[..., 1] * v2[..., 2] - v1[..., 2] * v2[..., 1],
                              v1[..., 2] * v2[..., 0] - v1[..., 0] * v2[..., 2],
                              v1[..., 0] * v2[..., 1] - v1[..., 1] * v2[..., 0]], dim=-1)
    cross_prod_norm = torch.norm(cross_prod, dim=-1)
    dot_prod = torch.sum(v1 * v2, dim=-1)

    return torch.atan2(cross_prod_norm, dot_prod)


def normal_redirect(points, normals, view_point):
    '''
    Make direction of normals towards the view point
    '''
    vec_dot = np.sum((view_point - points) * normals, axis=-1)
    mask = (vec_dot < 0.)
    redirected_normals = normals.copy()
    redirected_normals[mask] *= -1.
    return redirected_normals


def calculate_curvature_pca(points, neighbors, eps=1e-8):
    """
    :param points: B, N, C
    :param neighbors: N, N, S, C
    :param eps:
    :return:
    """
    # Calculate covariance matrices
    num_neighbors = neighbors.shape[-2]
    centered_neighbor_points = neighbors - points.unsqueeze(-2)
    cov_matrices = torch.matmul(centered_neighbor_points.transpose(-2, -1), centered_neighbor_points) / num_neighbors

    # Calculate eigenvalues and curvatures
    eigenvalues = torch.linalg.eigvalsh(cov_matrices)
    max_eigenvalues = eigenvalues.max(dim=-1)[0]
    curvatures = 2 * max_eigenvalues / (eigenvalues.sum(dim=-1) + eps)

    return curvatures


def index_gather(points, idx):
    """
    Input:
        points: input feats semdata, [B, N, C]
        idx: sample index semdata, [B, S, K]
    Return:
        new_points:, indexed feats semdata, [B, S, K, C]
    """
    dim = points.size(-1)
    n_clu = idx.size(1)
    # device = points.device
    view_list = list(idx.shape)
    view_len = len(view_list)
    # feats_shape = view_list
    xyz_shape = [-1] * (view_len + 1)
    xyz_shape[-1] = dim
    feats_shape = [-1] * (view_len + 1)
    feats_shape[1] = n_clu
    batch_indices = idx.unsqueeze(-1).expand(xyz_shape)
    points = points.unsqueeze(1).expand(feats_shape)
    new_points = torch.gather(points, dim=-2, index=batch_indices)
    return new_points


def query_ball_point(radius, nsample, xyz, new_xyz, itself_indices=None):
    """ Grouping layer in PointNet++.

    Inputs:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, (B, N, C)
        new_xyz: query points, (B, S, C)
        itself_indices (Optional): Indices of new_xyz into xyz (B, S).
          Used to try and prevent grouping the point itself into the neighborhood.
          If there is insufficient points in the neighborhood, or if left is none, the resulting cluster will
          still contain the center point.
    Returns:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])  # (B, S, N)
    sqrdists = torch.cdist(new_xyz, xyz)

    if itself_indices is not None:
        # Remove indices of the center points so that it will not be chosen
        batch_indices = torch.arange(B, dtype=torch.long).to(device)[:, None].repeat(1, S)  # (B, S)
        row_indices = torch.arange(S, dtype=torch.long).to(device)[None, :].repeat(B, 1)  # (B, S)
        group_idx[batch_indices, row_indices, itself_indices] = N

    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    if itself_indices is not None:
        group_first = itself_indices[:, :, None].repeat([1, 1, nsample])
    else:
        group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx.clip(min=0, max=N)


def calculate_curvature_pca_ball(queries, refs, num_neighbors=10, radius=0.1, eps=1e-8):
    # batch_size, num_points, num_features = points.size()
    idx = query_ball_point(radius, num_neighbors, refs, queries)  # [B, K, n_sampling]
    mean_node = torch.mean(queries, dim=-2, keepdim=True)
    cat_points = torch.cat([refs, mean_node], dim=1)
    os = torch.ones((refs.shape[0], refs.shape[1])).to(refs)
    neighbor_points = index_gather(cat_points, idx)  # [B, n_point, n_sample, 3]
    cat_os = torch.cat([os, torch.zeros_like(os[:, :1])], dim=-1).unsqueeze(-1)
    neighbor_os = index_gather(cat_os, idx).squeeze(-1)
    # Calculate covariance matrices
    inners = torch.sum(neighbor_os, dim=-1, keepdim=True)
    # w_neighbor_points = torch.einsum('bnkd,bnk->bnkd', neighbor_points, neighbor_os) / inners.unsqueeze(-1)
    centered_neighbor_points = neighbor_points - queries.unsqueeze(2)
    w_centered_neighbor_points = torch.einsum(
        'bnkd,bnk->bnkd', centered_neighbor_points, neighbor_os) / inners.unsqueeze(-1)
    cov_matrices = torch.matmul(centered_neighbor_points.transpose(-2, -1), w_centered_neighbor_points)
    # Calculate eigenvalues and curvatures
    eigenvalues = torch.linalg.eigvalsh(cov_matrices + eps)

    return eigenvalues


def log_boltzmann_kernel(cost, u, v, epsilon):
    kernel = (-cost + u.unsqueeze(-1) + v.unsqueeze(-2)) / epsilon
    return kernel


def sinkhorn(cost, p=None, q=None, epsilon=1e-4, thresh=1e-2, max_iter=100):
    if p is None or q is None:
        batch_size, num_x, num_y = cost.shape
        device = cost.device
        if p is None:
            p = torch.empty(batch_size, num_x, dtype=torch.float,
                            requires_grad=False, device=device).fill_(1.0 / num_x).squeeze()
        if q is None:
            q = torch.empty(batch_size, num_y, dtype=torch.float,
                            requires_grad=False, device=device).fill_(1.0 / num_y).squeeze()
    # Initialise approximation vectors in log domain
    u = torch.zeros_like(p).to(p)
    v = torch.zeros_like(q).to(q)
    # Stopping criterion, sinkhorn iterations
    for i in range(max_iter):
        u0, v0 = u, v
        # u^{l+1} = a / (K v^l)
        K = log_boltzmann_kernel(cost, u, v, epsilon)
        u_ = torch.log(p + 1e-8) - torch.logsumexp(K, dim=-1)
        u = epsilon * u_ + u
        # v^{l+1} = b / (K^T u^(l+1))
        Kt = log_boltzmann_kernel(cost, u, v, epsilon).transpose(-2, -1)
        v_ = torch.log(q + 1e-8) - torch.logsumexp(Kt, dim=-1)
        v = epsilon * v_ + v
        # Size of the change we have performed on u
        diff = torch.sum(torch.abs(u - u0), dim=-1) + torch.sum(torch.abs(v - v0), dim=-1)
        mean_diff = torch.mean(diff)
        if mean_diff < thresh:
            break
    # Transport plan pi = diag(a)*K*diag(b)
    K = log_boltzmann_kernel(cost, u, v, epsilon)
    gamma = torch.exp(K)
    # Sinkhorn distance
    return gamma, K
