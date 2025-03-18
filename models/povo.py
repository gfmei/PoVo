import torch
import torch.nn.functional as F


def farthest_point_sampling(pts, sample_k):
    """
    Performs Farthest Point Sampling on pts (N, 3) to pick sample_k points.
    Returns indices of the chosen points, shape: (sample_k,).

    Naive O(N * sample_k) approach using iterative distance updates.
    If N <= sample_k, simply returns all indices.

    Args:
        pts (torch.Tensor): (N, 3) set of points.
        sample_k (int): Number of samples to pick.

    Returns:
        selected_idx (torch.LongTensor): (sample_k,) farthest point sample indices.
    """
    device = pts.device
    N = pts.shape[0]

    # If there are fewer than sample_k points, return them all
    if N <= sample_k:
        return torch.arange(N, device=device, dtype=torch.long)

    # Initialize an array to store the minimum distance to any selected point
    distances = torch.full((N,), float('inf'), device=device)

    # Pick an initial point randomly
    first_idx = torch.randint(0, N, (1,), device=device)
    selected_idx = [first_idx.item()]

    # Update distances to the first selected point
    dist_to_first = torch.norm(pts - pts[first_idx], dim=1)
    distances = torch.min(distances, dist_to_first)

    # Iteratively pick next farthest point
    for _ in range(1, sample_k):
        # farthest point is one with max distance from the current set
        farthest_idx = torch.argmax(distances)
        selected_idx.append(farthest_idx.item())

        # Update the distances (minimum distance to any selected point)
        dist_to_new = torch.norm(pts - pts[farthest_idx], dim=1)
        distances = torch.min(distances, dist_to_new)

    return torch.tensor(selected_idx, device=device, dtype=torch.long)


def construct_superpoint_adjacency_graph_radius(
        pcds,  # (N, 3) tensor of point coordinates.
        spt_labels,  # (N,) tensor of superpoint labels (0,1,...).
        masks_list,  # List of masks for each frame, shape (num_masks, H, W) per frame.
        mapping_list,  # List of mappings for each frame, shape (N, 4) (visibility, y, x, z).
        sam_scores_list,  # List of lists of confidence scores for each frame.
        sample_k=10,  # Number of representative points per superpoint.
        min_points=5,  # Minimum points for a valid superpoint.
        radius_threshold=0.2,  # Radius threshold to connect two superpoints if min-dist < this value.
        point_feas=None,  # (N, F) optional tensor of point features.
        mask_thres=0.2,  # Threshold for mask contribution.
        sim_thres=0.5,  # Threshold for feature similarity.
        n_mask=5,  # Maximum number of masks to consider per superpoint pair.
        batch_size=32  # Batch size for chunking distance computations.
):
    """
    Constructs a superpoint adjacency graph using a radius threshold:

    1. Sample up to sample_k points from each superpoint (representatives).
    2. For each pair of superpoints (i, j), compute the minimum Euclidean distance among
       their representatives. If < radius_threshold, create an edge with weight = exp(-dist).
    3. Modulate edge weights by mask-overlap and (optionally) feature similarity, as before.
    4. Compute superpoint confidence scores from mask coverage.

    Returns:
        final_adjacency (torch.Tensor): (M, M) adjacency among valid superpoints.
        superpoint_confidence_scores (torch.Tensor): (M,) confidence scores for valid superpoints.
    """
    device = pcds.device
    spt_labels = spt_labels.to(device).long()

    # --- Step 1: Identify valid superpoints ---
    unique_spts, sp_counts = torch.unique(spt_labels, return_counts=True)
    valid_mask = (sp_counts >= min_points)
    if not valid_mask.any():
        raise ValueError(f"No superpoints have at least {min_points} points.")

    valid_superpoints = unique_spts[valid_mask]  # shape: [M]
    M = valid_superpoints.size(0)

    # --- Step 2: Representative sampling ---
    reps_list = []
    for sp_id in valid_superpoints:
        idx = torch.nonzero(spt_labels == sp_id, as_tuple=False).view(-1)
        n_points_sp = idx.numel()
        if n_points_sp > sample_k:
            # perm = torch.randperm(n_points_sp, device=device)[:sample_k]
            # rep_idx = idx[perm]
            # Gather all points in this superpoint
            sp_points = pcds[idx]  # shape: (n_points_sp, 3)
            # Farthest point sampling
            chosen_idx_local = farthest_point_sampling(sp_points, sample_k)  # local indices in [0..n_points_sp-1]
            rep_idx = idx[chosen_idx_local]

        else:
            rep_idx = idx
            # Pad if needed so each SP always has sample_k reps
            if n_points_sp < sample_k:
                pad = rep_idx[-1].repeat(sample_k - n_points_sp)
                rep_idx = torch.cat([rep_idx, pad], dim=0)
        reps_list.append(pcds[rep_idx].unsqueeze(0))  # shape: (1, sample_k, 3)
    # reps: (M, sample_k, 3)
    reps = torch.cat(reps_list, dim=0)

    # --- Step 3: Compute all pairwise min-distances (chunked to avoid O(M^2*K^2) memory blow-up) ---
    # We'll build a full (M, M) distance matrix, storing the minimum among the K*K pairs.
    all_dists = torch.empty((M, M), device=device)

    for i in range(M):
        d_row = torch.empty((M,), device=device)
        # Only need to compute from i..M, then symmetrize
        for start in range(i, M, batch_size):
            end = min(M, start + batch_size)
            # reps for superpoints in [start, end)
            B = reps[start:end]  # (batch_size, sample_k, 3)
            A_expanded = reps[i].unsqueeze(0).expand(end - start, -1, -1)  # (batch_size, sample_k, 3)

            # cdist => shape: (batch_size, sample_k, sample_k)
            D = torch.cdist(A_expanded, B)
            d_min_batch = D.view(end - start, -1).min(dim=1)[0]
            d_row[start:end] = d_min_batch

        # self-distance to inf
        d_row[i] = 1
        all_dists[i] = d_row
        # Symmetrize
        if i > 0:
            all_dists[:i, i] = all_dists[i, :i]

    # --- Step 4: Build adjacency by radius threshold: adjacency = exp(-dists) if dist < radius_threshold ---
    adjacency = torch.zeros((M, M), device=device)
    # Boolean for which edges are valid
    within_radius = (all_dists < radius_threshold)
    # adjacency[i,j] = exp(-dist(i,j)) if within radius
    adjacency[within_radius] = torch.sigmoid(-all_dists[within_radius])

    # Force diagonal to 0 (no self-edges)
    adjacency.fill_diagonal_(0.0)

    # --- Step 5: Aggregate mask coverage per superpoint ---
    num_masks_total = sum(m.shape[0] for m in masks_list)
    N = pcds.shape[0]
    mask_coverage = torch.zeros((N, num_masks_total), dtype=torch.bool, device=device)
    mask_confidences = torch.zeros((num_masks_total,), dtype=torch.float, device=device)

    offset = 0
    for masks, mapping, sam_scores in zip(masks_list, mapping_list, sam_scores_list):
        masks = masks.to(device)
        mapping = mapping.to(device)
        num_masks, H, W = masks.shape

        # Points that are visible in this frame
        vis = (mapping[:, 3] == 1)
        vis_idx = torch.where(vis)[0]
        y_coords = mapping[vis_idx, 1].long().clamp(0, H - 1)
        x_coords = mapping[vis_idx, 2].long().clamp(0, W - 1)

        point_masks = masks[:, y_coords, x_coords].permute(1, 0)  # (num_vis, num_masks)
        mask_coverage[vis_idx, offset:offset + num_masks] = point_masks

        sam_scores_tensor = torch.as_tensor(sam_scores, dtype=torch.float, device=device)
        mask_confidences[offset:offset + num_masks] = sam_scores_tensor
        offset += num_masks

    # Group coverage by superpoint
    _, inverse_indices_all = torch.unique(spt_labels, return_inverse=True)
    sp_counts_all = sp_counts.unsqueeze(1).float()  # (num_superpoints_total, 1)
    mask_coverage_grouped = torch.zeros(
        (unique_spts.shape[0], num_masks_total), dtype=torch.long, device=device
    )
    mask_coverage_grouped.index_add_(
        0, inverse_indices_all, mask_coverage.long()
    )
    mask_fraction = mask_coverage_grouped.float() / (sp_counts_all + 1e-6)

    # Binary: does superpoint have coverage >= mask_thres?
    mask_contribution = (mask_fraction >= mask_thres)

    # Filter to valid superpoints
    sp_masks_valid = mask_contribution[valid_mask]  # (M, num_masks_total)
    sp_masks_raw_valid = mask_fraction[valid_mask]  # (M, num_masks_total)

    # --- Step 6: (Optional) superpoint features ---
    if point_feas is not None:
        point_feas = point_feas.to(device)
        F_dim = point_feas.shape[1]

        sp_features_all = torch.zeros((unique_spts.shape[0], F_dim), device=device)
        sp_features_all.index_add_(0, inverse_indices_all, point_feas)
        sp_features_all /= (sp_counts_all + 1e-6)

        sp_features_valid = sp_features_all[valid_mask]  # (M, F_dim)
        sp_features_valid = F.normalize(sp_features_valid, p=2, dim=1)  # for cosine similarity
    else:
        sp_features_valid = None

    # --- Step 7: Modulate adjacency by mask overlap and feature similarity ---
    # We'll loop over pairs (i, j) where adjacency[i,j] > 0, compute overlap ratio and feature sim,
    # then multiply adjacency[i,j] by (overlap * feature_sim).

    # To avoid huge memory usage (M x M x num_masks), we do this in row-chunks:
    chunk_size = batch_size  # you can adjust

    for start_i in range(0, M, chunk_size):
        end_i = min(M, start_i + chunk_size)
        for i in range(start_i, end_i):
            # Indices j where there's a positive edge
            row_vals = adjacency[i]  # shape: (M,)
            j_indices = torch.nonzero(row_vals > 0, as_tuple=True)[0]
            if j_indices.numel() == 0:
                continue

            # sp_masks_valid[i]: shape (num_masks_total,)
            sp_mask_i = sp_masks_valid[i].unsqueeze(0)  # (1, num_masks_total)
            # We gather all neighbors' masks
            sp_mask_j = sp_masks_valid[j_indices]  # (num_neighbors, num_masks_total)

            # Union over each neighbor
            union = (sp_mask_i | sp_mask_j).float()  # (num_neighbors, num_masks_total)

            # We only want top-n_mask columns for each row in union
            # shape: (num_neighbors, num_masks_total)
            n_mask_eff = min(n_mask, num_masks_total)
            # Flatten for topk
            union_flat = union.view(sp_mask_j.size(0), -1)
            _, topk_inds = union_flat.topk(k=n_mask_eff, dim=1)

            sp_mask_i_expanded = sp_mask_i.expand(sp_mask_j.size(0), -1)  # (num_neighbors, num_masks_total)

            # Gather top-n_mask columns from i and j
            sp_top_masks_i = torch.gather(sp_mask_i_expanded, 1, topk_inds)
            sp_top_masks_j = torch.gather(sp_mask_j, 1, topk_inds)

            # Intersection & union within these top masks
            inter = (sp_top_masks_i & sp_top_masks_j).sum(dim=1).float()
            uni = (sp_top_masks_i | sp_top_masks_j).sum(dim=1).float()
            overlap_ratios = inter / (uni + 1e-6)

            # Feature similarity if available
            if sp_features_valid is not None:
                # cos sim => (sp_features_valid[i] * sp_features_valid[j]).sum(dim=-1)
                feat_i = sp_features_valid[i]  # (F_dim,)
                feat_j = sp_features_valid[j_indices]  # (num_neighbors, F_dim)
                dot_prod = torch.sum(feat_i * feat_j, dim=1)  # (num_neighbors,)
                # Map [-1,1] -> [0,1]
                sim_vals = (dot_prod + 1) / 2
                sim_vals = torch.clamp(sim_vals, 0.0, 1.0)
                # Where sim < sim_thres, set to 1
                sim_vals = torch.where(sim_vals < sim_thres, torch.ones_like(sim_vals), sim_vals)
            else:
                sim_vals = torch.ones_like(overlap_ratios)

            # Final modulation
            mod_factor = overlap_ratios * sim_vals

            # Multiply adjacency
            adjacency[i, j_indices] = adjacency[i, j_indices] * mod_factor
            # adjacency[i, j_indices] = torch.max(torch.sign(adjacency[i, j_indices])*mod_factor, adjacency[i, j_indices])
            adjacency[j_indices, i] = adjacency[i, j_indices]  # keep symmetry

    # --- Step 8: Superpoint confidence scores ---
    # Weighted by SAM confidences
    sp_mask_weights = sp_masks_raw_valid * mask_confidences.unsqueeze(0)  # (M, num_masks_total)
    superpoint_mask_confidences = sp_mask_weights.sum(dim=1)
    total_mask_contributions = sp_masks_raw_valid.sum(dim=1) + 1e-6
    superpoint_confidence_scores = superpoint_mask_confidences / total_mask_contributions
    superpoint_confidence_scores = torch.clamp(superpoint_confidence_scores, min=0.0)

    return adjacency, superpoint_confidence_scores


def construct_superpoint_adjacency_graph_radius(
        pcds,  # (N, 3) tensor of point coordinates.
        spt_labels,  # (N,) tensor of superpoint labels (0,1,...).
        masks_list,  # List of masks for each frame, shape (num_masks, H, W) per frame.
        mapping_list,  # List of mappings for each frame, shape (N, 4) (visibility, y, x, z).
        sam_scores_list,  # List of lists of confidence scores for each frame.
        sample_k=10,  # Number of representative points per superpoint.
        min_points=5,  # Minimum points for a valid superpoint.
        radius_threshold=0.2,  # Radius threshold to connect two superpoints if min-dist < this value.
        point_feas=None,  # (N, F) optional tensor of point features.
        mask_thres=0.2,  # Threshold for mask contribution.
        sim_thres=0.5,  # Threshold for feature similarity.
        n_mask=5,  # Maximum number of masks to consider per superpoint pair.
        top_k=10,  # For each mask, only keep top_k superpoints by coverage fraction.
        batch_size=32  # Batch size for chunking distance computations.
):
    """
    Constructs a superpoint adjacency graph using a radius threshold, with:
      - Farthest Point Sampling (FPS) for representatives.
      - For each mask, only keep the top_k superpoints by coverage fraction.

    1. Identify valid superpoints (>= min_points).
    2. Sample up to sample_k points from each superpoint using FPS.
    3. Compute pairwise min-distances between superpoints' reps -> radius-based adjacency.
    4. Aggregate mask coverage. For each mask, keep only top_k superpoints (zero out others).
    5. Build mask_contribution using mask_thres.
    6. (Optional) compute features -> modulate adjacency by overlap & feature sim.
    7. Compute superpoint confidence scores from coverage fractions.

    Returns:
        final_adjacency (torch.Tensor): (M, M) adjacency among valid superpoints.
        superpoint_confidence_scores (torch.Tensor): (M,) confidence scores for valid superpoints.
    """
    device = pcds.device
    spt_labels = spt_labels.to(device).long()

    # ================ STEP 1: Identify valid superpoints ================
    unique_spts, sp_counts = torch.unique(spt_labels, return_counts=True)
    valid_mask = (sp_counts >= min_points)
    if not valid_mask.any():
        raise ValueError(f"No superpoints have at least {min_points} points.")

    valid_superpoints = unique_spts[valid_mask]  # shape: [M]
    M = valid_superpoints.size(0)

    # ================ STEP 2: Representative sampling (FPS) ================
    reps_list = []
    for sp_id in valid_superpoints:
        idx = torch.nonzero(spt_labels == sp_id, as_tuple=False).view(-1)
        n_points_sp = idx.numel()
        if n_points_sp > sample_k:
            # perm = torch.randperm(n_points_sp, device=device)[:sample_k]
            # rep_idx = idx[perm]
            # Gather all points in this superpoint
            sp_points = pcds[idx]  # shape: (n_points_sp, 3)
            # Farthest point sampling
            chosen_idx_local = farthest_point_sampling(sp_points, sample_k)  # local indices in [0..n_points_sp-1]
            rep_idx = idx[chosen_idx_local]

        else:
            rep_idx = idx
            # Pad if needed so each SP always has sample_k reps
            if n_points_sp < sample_k:
                pad = rep_idx[-1].repeat(sample_k - n_points_sp)
                rep_idx = torch.cat([rep_idx, pad], dim=0)
        reps_list.append(pcds[rep_idx].unsqueeze(0))  # shape: (1, sample_k, 3)
    # Reps: (M, <=sample_k, 3)
    reps = torch.cat(reps_list, dim=0)

    # ================ STEP 3: Compute pairwise min-distances (chunked) ================
    all_dists = torch.empty((M, M), device=device)

    for i in range(M):
        d_row = torch.empty((M,), device=device)
        A = reps[i]  # shape: (Ki, 3), Ki <= sample_k
        for start in range(i, M, batch_size):
            end = min(M, start + batch_size)
            B = reps[start:end]  # shape: (batch_size, Kj, 3)

            d_min_batch = []
            for b_idx in range(B.shape[0]):
                # cdist => shape (Ki, Kj)
                Dij = torch.cdist(A, B[b_idx])
                d_min_batch.append(Dij.min().unsqueeze(0))
            d_min_batch = torch.cat(d_min_batch, dim=0)  # shape: (batch_size,)

            d_row[start:end] = d_min_batch

        d_row[i] = float("inf")  # self-distance
        all_dists[i] = d_row
        # Symmetrize
        if i > 0:
            all_dists[:i, i] = all_dists[i, :i]

    # ================ STEP 4: Radius-based adjacency ================
    adjacency = torch.zeros((M, M), device=device)
    within_radius = (all_dists < radius_threshold)
    adjacency[within_radius] = torch.exp(-all_dists[within_radius])
    adjacency.fill_diagonal_(0.0)

    # ================ STEP 5: Aggregate mask coverage ================
    num_masks_total = sum(m.shape[0] for m in masks_list)
    N = pcds.shape[0]
    mask_coverage = torch.zeros((N, num_masks_total), dtype=torch.bool, device=device)
    mask_confidences = torch.zeros((num_masks_total,), dtype=torch.float, device=device)

    offset = 0
    for masks, mapping, sam_scores in zip(masks_list, mapping_list, sam_scores_list):
        masks = masks.to(device)
        mapping = mapping.to(device)
        num_masks, H, W = masks.shape

        # Points that are "visible" in this frame
        vis = (mapping[:, 3] == 1)
        vis_idx = torch.where(vis)[0]
        y_coords = mapping[vis_idx, 1].long().clamp(0, H - 1)
        x_coords = mapping[vis_idx, 2].long().clamp(0, W - 1)

        point_masks = masks[:, y_coords, x_coords].permute(1, 0)  # (num_vis, num_masks)
        mask_coverage[vis_idx, offset:offset + num_masks] = point_masks

        # Store SAM scores
        sam_scores_tensor = torch.as_tensor(sam_scores, dtype=torch.float, device=device)
        mask_confidences[offset:offset + num_masks] = sam_scores_tensor
        offset += num_masks

    # Group coverage by superpoint (including invalid ones, then filter later)
    _, inverse_indices_all = torch.unique(spt_labels, return_inverse=True)
    sp_counts_all = sp_counts.unsqueeze(1).float()  # (num_superpoints_total, 1)

    # (num_superpoints_total, num_masks_total)
    mask_coverage_grouped = torch.zeros(
        (unique_spts.shape[0], num_masks_total), dtype=torch.long, device=device
    )
    mask_coverage_grouped.index_add_(0, inverse_indices_all, mask_coverage.long())

    # coverage fraction: fraction of each superpoint's points covered by each mask
    mask_fraction = mask_coverage_grouped.float() / (sp_counts_all + 1e-6)

    # ---------------- STEP 5.2: Keep only top_k superpoints per mask ----------------
    #  -- (Optional) If you want to rank by coverage * mask_confidence, you'd do:
    # score_matrix = mask_fraction * mask_confidences.unsqueeze(0)
    # and then sort by score_matrix instead. For now, we just use 'mask_fraction'.

    # Sort each mask (each column) in descending order
    sorted_vals, sorted_indices = torch.sort(mask_fraction, dim=0, descending=True)
    # Create a boolean mask
    topk_mask = torch.zeros_like(mask_fraction, dtype=torch.bool)

    # Mark True for the top_k superpoints in each mask's column
    for m in range(mask_fraction.shape[1]):
        # Indices of top_k rows for mask m
        topk_rows = sorted_indices[:top_k, m]
        topk_mask[topk_rows, m] = True

    # Zero out coverage fraction for superpoints not in top_k for this mask
    mask_fraction = torch.where(topk_mask, mask_fraction, torch.zeros_like(mask_fraction))

    # ================ STEP 6: Build mask_contribution & filter valid superpoints ================
    # Binary: does superpoint have coverage >= mask_thres (but only among top_k now!)
    mask_contribution = (mask_fraction >= mask_thres)

    # Filter to valid superpoints
    sp_masks_valid = mask_contribution[valid_mask]  # (M, num_masks_total)
    sp_masks_raw_valid = mask_fraction[valid_mask]  # (M, num_masks_total)

    # ================ (Optional) Superpoint features ================
    if point_feas is not None:
        point_feas = point_feas.to(device)
        F_dim = point_feas.shape[1]

        sp_features_all = torch.zeros((unique_spts.shape[0], F_dim), device=device)
        sp_features_all.index_add_(0, inverse_indices_all, point_feas)
        sp_features_all /= (sp_counts_all + 1e-6)

        # Normalize for cosine similarity
        sp_features_valid = F.normalize(sp_features_all[valid_mask], p=2, dim=1)
    else:
        sp_features_valid = None

    # ================ STEP 7: Modulate adjacency by mask overlap & optional feature sim ================
    chunk_size = batch_size
    for start_i in range(0, M, chunk_size):
        end_i = min(M, start_i + chunk_size)
        for i in range(start_i, end_i):
            row_vals = adjacency[i]  # shape: (M,)
            j_indices = torch.nonzero(row_vals > 0, as_tuple=True)[0]
            if j_indices.numel() == 0:
                continue

            sp_mask_i = sp_masks_valid[i].unsqueeze(0)  # (1, num_masks_total)
            sp_mask_j = sp_masks_valid[j_indices]  # (num_neighbors, num_masks_total)

            union = (sp_mask_i | sp_mask_j).float()
            n_mask_eff = min(n_mask, union.shape[1])

            # Flatten for topk
            union_flat = union.view(sp_mask_j.size(0), -1)
            _, topk_inds = union_flat.topk(k=n_mask_eff, dim=1)

            sp_mask_i_expanded = sp_mask_i.expand(sp_mask_j.size(0), -1)
            sp_top_masks_i = torch.gather(sp_mask_i_expanded, 1, topk_inds)
            sp_top_masks_j = torch.gather(sp_mask_j, 1, topk_inds)

            inter = (sp_top_masks_i & sp_top_masks_j).sum(dim=1).float()
            uni = (sp_top_masks_i | sp_top_masks_j).sum(dim=1).float()
            overlap_ratios = inter / (uni + 1e-6)

            # Feature similarity (if available)
            if sp_features_valid is not None:
                feat_i = sp_features_valid[i]
                feat_j = sp_features_valid[j_indices]
                dot_prod = torch.sum(feat_i * feat_j, dim=1)
                sim_vals = (dot_prod + 1.0) / 2.0
                sim_vals = torch.clamp(sim_vals, 0.0, 1.0)
                # Where similarity < sim_thres, set factor to 1
                sim_vals = torch.where(sim_vals < sim_thres,
                                       torch.ones_like(sim_vals),
                                       sim_vals)
            else:
                sim_vals = torch.ones_like(overlap_ratios)

            mod_factor = overlap_ratios * sim_vals

            # adjacency[i, j_indices] = adjacency[i, j_indices] * mod_factor
            adjacency[i, j_indices] = torch.max(mod_factor, adjacency[i, j_indices])
            adjacency[j_indices, i] = adjacency[i, j_indices]  # symmetric

    # ================ STEP 8: Superpoint confidence scores ================
    # Weighted by SAM confidences
    sp_mask_weights = sp_masks_raw_valid * mask_confidences.unsqueeze(0)
    superpoint_mask_confidences = sp_mask_weights.sum(dim=1)
    total_mask_contributions = sp_masks_raw_valid.sum(dim=1) + 1e-6
    superpoint_confidence_scores = superpoint_mask_confidences / total_mask_contributions
    superpoint_confidence_scores = torch.clamp(superpoint_confidence_scores, min=0.0)

    return adjacency, superpoint_confidence_scores
