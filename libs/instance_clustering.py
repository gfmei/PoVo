"""Multi-view evidence and constrained spectral instance clustering."""

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
import warnings

from libs import lib_cluster
from libs.lib_graph import farthest_point_sample, index_points


def kmeans(x, n_clusters, iters=20, weights=None):
    """Lloyd rounding with FPS/seeded k-means++ starts and stable-label stopping.

    Empty clusters retain their centroids instead of moving to the origin.
    """
    if not 1 <= n_clusters <= x.shape[1] or iters < 1:
        raise ValueError('invalid cluster count or iterations')
    if weights is None:
        weights = torch.ones(x.shape[:2], device=x.device, dtype=x.dtype)
    if weights.shape != x.shape[:2] or not torch.isfinite(weights).all() or (weights < 0).any() or (weights.sum(1) <= 0).any():
        raise ValueError('weights must be finite, nonnegative, and match the observations')
    best_loss = torch.full((x.shape[0],), float('inf'), device=x.device)
    best_gamma, best_centroids = None, None
    for start in range(4):
        if start == 0:
            centroids = index_points(x, farthest_point_sample(x, n_clusters, is_center=True))
        else:
            generator = torch.Generator(device=x.device).manual_seed(start)
            chosen = torch.multinomial(weights, 1, generator=generator)
            centroids = index_points(x, chosen)
            for _ in range(1, n_clusters):
                distances = torch.cdist(x, centroids).square().amin(-1)
                probability = distances * weights
                probability = torch.where(probability.sum(1, keepdim=True) > 0, probability, weights)
                chosen = torch.multinomial(probability, 1, generator=generator)
                centroids = torch.cat((centroids, index_points(x, chosen)), dim=1)
        previous = None
        for _ in range(iters):
            distance = torch.cdist(x, centroids).square()
            labels = distance.argmin(-1)
            gamma = F.one_hot(labels, n_clusters).to(x)
            mass = gamma * weights[..., None]
            totals = mass.sum(1)
            updated = mass.transpose(-1, -2) @ x / totals.clamp_min(1e-9)[..., None]
            centroids = torch.where((totals > 0)[..., None], updated, centroids)
            if previous is not None and torch.equal(labels, previous):
                break
            previous = labels
        distance = torch.cdist(x, centroids).square()
        loss = (distance.gather(-1, labels[..., None]).squeeze(-1) * weights).sum(1)
        better = loss < best_loss
        if best_gamma is None:
            best_gamma, best_centroids = gamma.clone(), centroids.clone()
        else:
            best_gamma[better], best_centroids[better] = gamma[better], centroids[better]
        best_loss = torch.minimum(best_loss, loss)
    return best_gamma, best_gamma.mean(1), best_centroids


@dataclass
class FrameEvidence:
    agreement: torch.Tensor
    cannot_link: torch.Tensor
    positive_votes: torch.Tensor
    negative_votes: torch.Tensor
    confidence: torch.Tensor
    spatial_support: torch.Tensor | None = None


def frame_mask_evidence(labels, masks_list, mappings, scores_list, geometry,
                        mask_threshold=.3, minimum_visibility=.1,
                        conflict_confidence=.8, minimum_conflicts=2, minimum_conflict_fraction=0.):
    """One normalized vote per frame, plus repeated exclusive-mask constraints.

    Strong conflicting masks must have 2D IoU below .1. Unknown and occluded
    pairs cast no vote. Duplicating a mask inside a frame changes no vote.
    """
    if not (len(masks_list) == len(mappings) == len(scores_list)):
        raise ValueError('each frame requires masks, mappings, and scores')
    if not 0 <= minimum_conflict_fraction <= 1:
        raise ValueError('minimum conflict fraction must lie between zero and one')
    device = labels.device
    _, labels, counts = torch.unique(labels, return_inverse=True, return_counts=True)
    count = counts.numel()
    rows, columns = torch.where(torch.triu(geometry > 0, diagonal=1))
    edge_numerator = torch.zeros(len(rows), device=device)
    edge_denominator = torch.zeros_like(edge_numerator)
    edge_conflicts = torch.zeros_like(edge_numerator)
    edge_observations = torch.zeros_like(edge_numerator)
    positive = torch.zeros((count, count), device=device)
    negative = torch.zeros_like(positive)
    confidence_sum = torch.zeros(count, device=device)
    confidence_count = torch.zeros_like(confidence_sum)
    for masks, mapping, scores in zip(masks_list, mappings, scores_list):
        if len(masks) == 0:
            continue
        masks, mapping = masks.to(device).bool(), mapping.to(device)
        scores = torch.nan_to_num(torch.as_tensor(scores, device=device).float(), nan=0., posinf=0., neginf=0.).clamp(0, 1)
        visible = torch.where(mapping[:, 3] == 1)[0]
        visible_count = torch.bincount(labels[visible], minlength=count)
        observed = visible_count >= (minimum_visibility * counts).clamp_min(1)
        y = mapping[visible, 1].long().clamp(0, masks.shape[1] - 1)
        x = mapping[visible, 2].long().clamp(0, masks.shape[2] - 1)
        coverage = torch.zeros((count, len(masks)), device=device, dtype=torch.long)
        coverage.index_add_(0, labels[visible], masks[:, y, x].T.long())
        fraction = coverage.float() / visible_count.clamp_min(1)[:, None]
        membership = (fraction >= mask_threshold) & observed[:, None]
        strength = membership * scores[None, :]
        best_confidence, best_mask = strength.max(1)
        frame_supported = best_confidence > 0
        confidence_sum += best_confidence
        confidence_count += frame_supported
        flat = masks.flatten(1).float()
        intersection = flat @ flat.T
        areas = flat.sum(1)
        mask_iou = intersection / (areas[:, None] + areas[None, :] - intersection).clamp_min(1)
        reliable = frame_supported & (best_confidence >= conflict_confidence)
        pair_reliable = reliable[:, None] & reliable[None, :]
        positive += ((best_mask[:, None] == best_mask[None, :]) & pair_reliable).float()
        negative += ((mask_iou[best_mask[:, None], best_mask[None, :]] < .1) & pair_reliable).float()
        for start in range(0, len(rows), 4096):
            i, j = rows[start:start + 4096], columns[start:start + 4096]
            shared = (membership[i] & membership[j]) * scores
            shared_confidence = shared.max(1).values
            pair_confidence = torch.maximum(best_confidence[i], best_confidence[j])
            jointly_supported = observed[i] & observed[j] & frame_supported[i] & frame_supported[j]
            edge_numerator[start:start + len(i)] += torch.where(jointly_supported, shared_confidence, 0.)
            edge_denominator[start:start + len(i)] += torch.where(jointly_supported, pair_confidence, 0.)
            edge_observations[start:start + len(i)] += jointly_supported.float()
            exclusive_i = (membership[i] & ~membership[j]) * scores
            exclusive_j = (membership[j] & ~membership[i]) * scores
            score_i, mask_i = exclusive_i.max(1)
            score_j, mask_j = exclusive_j.max(1)
            conflict = jointly_supported & (score_i >= conflict_confidence) & (score_j >= conflict_confidence) & (mask_iou[mask_i, mask_j] < .1)
            edge_conflicts[start:start + len(i)] += conflict.float()
        del flat, coverage
    positive.fill_diagonal_(0)
    negative.fill_diagonal_(0)
    agreement = torch.zeros_like(geometry)
    ratios = edge_numerator / edge_denominator.clamp_min(1e-9)
    agreement[rows, columns] = agreement[columns, rows] = ratios
    # Global cannot-links protect against paths through intermediate nodes.
    cannot_link = (negative >= minimum_conflicts) & (negative >= positive)
    local_conflicts = (edge_conflicts >= minimum_conflicts) & (
        edge_conflicts >= minimum_conflict_fraction * edge_observations)
    cannot_link[rows[local_conflicts], columns[local_conflicts]] = True
    cannot_link[columns[local_conflicts], rows[local_conflicts]] = True
    cannot_link.fill_diagonal_(False)
    return FrameEvidence(agreement, cannot_link, positive, negative,
                         confidence_sum / confidence_count.clamp_min(1), geometry > 0)


def modulate_geometry(geometry, agreement, point_features, labels, similarity_threshold=.8):
    _, inverse, counts = torch.unique(labels, return_inverse=True, return_counts=True)
    features = torch.zeros((len(counts), point_features.shape[1]), device=geometry.device)
    features.index_add_(0, inverse, point_features.float())
    features = F.normalize(features / counts[:, None].clamp_min(1), dim=-1)
    similarity = ((features @ features.T + 1) / 2).clamp(0, 1)
    similarity = torch.where(similarity >= similarity_threshold, similarity, 0.)
    known = features.norm(dim=1) > 0
    similarity = torch.where(known[:, None] & known[None, :], similarity, 1.)
    result = geometry * agreement * similarity
    result.fill_diagonal_(0)
    return (result + result.T) / 2


def normalized_cut(adjacency, labels):
    """Mean cut/volume across occupied groups; unused IDs carry no reward."""
    labels = torch.unique(labels, return_inverse=True)[1]
    count = int(labels.max()) + 1
    membership = F.one_hot(labels, count).to(adjacency)
    volume = membership.T @ adjacency.sum(1)
    inside = (membership.T @ adjacency @ membership).diagonal()
    return ((volume - inside) / volume.clamp_min(1e-9)).mean()


def quality_spectral_labels(adjacency, evidence, budget=128):
    """Select local k with cut quality, view consistency, and perturbation stability.

    All coefficients and candidate rules are fixed before held-out evaluation.
    No ground-truth labels or performance metrics enter this selection.
    """
    components = lib_cluster.split_disconnected_clusters(adjacency, torch.zeros(len(adjacency), device=adjacency.device, dtype=torch.long))
    counts = torch.bincount(components)
    total = int(counts[counts > 1].sum())
    result = torch.arange(len(adjacency), device=adjacency.device)
    offset = len(adjacency)
    for component in torch.where(counts > 1)[0].tolist():
        indices = torch.where(components == component)[0]
        size = len(indices)
        maximum = min(size, max(2, (budget * size + total - 1) // total))
        local = adjacency[indices[:, None], indices[None, :]]
        degree = local.sum(1).clamp_min(1e-5)
        laplacian = torch.eye(size, device=local.device) - local / torch.sqrt(degree[:, None] * degree[None, :])
        eigenvalues, vectors = torch.linalg.eigh((laplacian + laplacian.T) / 2)
        gaps = eigenvalues[1:maximum] - eigenvalues[:maximum - 1]
        candidates = {1, 2}
        candidates.update(int(value) + 1 for value in gaps.topk(min(4, len(gaps))).indices.tolist())
        gap_k = int(gaps.argmax()) + 1
        candidates.update(value for value in (gap_k - 1, gap_k + 1) if 1 <= value <= maximum)
        generator = torch.Generator(device=local.device).manual_seed(0)
        perturbation = torch.rand(local.shape, device=local.device, generator=generator) * .1 + .95
        perturbation = (perturbation + perturbation.T) / 2
        perturbed = local * perturbation
        perturbed_degree = perturbed.sum(1).clamp_min(1e-5)
        perturbed_laplacian = torch.eye(size, device=local.device) - perturbed / torch.sqrt(perturbed_degree[:, None] * perturbed_degree[None, :])
        _, perturbed_vectors = torch.linalg.eigh((perturbed_laplacian + perturbed_laplacian.T) / 2)
        positives = evidence.positive_votes[indices[:, None], indices[None, :]]
        negatives = evidence.negative_votes[indices[:, None], indices[None, :]]
        weight = (positives + negatives).sum().clamp_min(1)
        best_labels, best_score = None, float('inf')
        for count in sorted(candidates):
            if count == 1:
                labels = torch.zeros(size, device=local.device, dtype=torch.long)
                instability = torch.tensor(0., device=local.device)
            else:
                embedding = F.normalize(vectors[:, :count], dim=-1).unsqueeze(0)
                changed_embedding = F.normalize(perturbed_vectors[:, :count], dim=-1).unsqueeze(0)
                labels = kmeans(embedding, count)[0].argmax(-1).squeeze(0)
                changed = kmeans(changed_embedding, count)[0].argmax(-1).squeeze(0)
                instability = ((labels[:, None] == labels[None, :]) != (changed[:, None] == changed[None, :])).float().mean()
            same = labels[:, None] == labels[None, :]
            disagreement = (negatives * same + positives * ~same).sum() / weight
            occupied = len(torch.unique(labels))
            score = .5 * normalized_cut(local, labels) + .5 * disagreement + .02 * (occupied - 1) + .1 * instability
            if float(score) < best_score:
                best_score, best_labels = float(score), labels
        result[indices] = best_labels + offset
        offset += int(best_labels.max()) + 1
    return torch.unique(result, return_inverse=True)[1]


def split_conflicting_clusters(labels, cannot_link, adjacency):
    """Greedy conflict coloring, followed by connected-component separation."""
    labels = labels.clone()
    offset = int(labels.max()) + 1
    for cluster in torch.unique(labels).tolist():
        indices = torch.where(labels == cluster)[0]
        conflicts = cannot_link[indices[:, None], indices[None, :]]
        if not conflicts.any():
            continue
        assigned = torch.full((len(indices),), -1, device=labels.device, dtype=torch.long)
        order = torch.argsort(conflicts.sum(1), descending=True, stable=True)
        for node in order.tolist():
            valid_colors = []
            for color in range(int(assigned.max()) + 1):
                members = assigned == color
                if not conflicts[node, members].any():
                    strength = float(adjacency[indices[node], indices[members]].sum())
                    valid_colors.append((strength, -color))
            color = -max(valid_colors)[1] if valid_colors else int(assigned.max()) + 1
            assigned[node] = color
        labels[indices] = assigned + offset
        offset += int(assigned.max()) + 1
    labels = torch.unique(labels, return_inverse=True)[1]
    return lib_cluster.split_disconnected_clusters(adjacency, labels)


def constrained_average_linkage(affinity, sizes, forbidden, threshold=.5):
    """Average linkage whose unions preserve every cannot-link constraint."""
    affinity, sizes, forbidden = affinity.clone(), sizes.float().clone(), forbidden.clone()
    count = len(sizes)
    parent = torch.arange(count, device=affinity.device)
    affinity.fill_diagonal_(float('-inf'))
    affinity[forbidden] = float('-inf')
    while count > 1:
        i, j = divmod(int(affinity.argmax()), count)
        if float(affinity[i, j]) <= threshold:
            break
        merged = (affinity[i] * sizes[i] + affinity[j] * sizes[j]) / (sizes[i] + sizes[j])
        blocked = forbidden[i] | forbidden[j]
        affinity[i, :], affinity[:, i] = merged, merged
        forbidden[i, :], forbidden[:, i] = blocked, blocked
        affinity[i, blocked], affinity[blocked, i] = float('-inf'), float('-inf')
        affinity[i, i] = float('-inf')
        affinity[j, :], affinity[:, j] = float('-inf'), float('-inf')
        sizes[i] += sizes[j]
        parent[parent == j] = i
    return torch.unique(parent, return_inverse=True)[1]


def constrained_edge_linkage(edge_sums, edge_counts, forbidden, threshold=.5):
    """Merge using accumulated edge evidence, preserving every cannot-link.

    Sum both evidence and effective pair counts after each union. Averaging
    already normalized affinities by cluster size would lose this information.
    """
    if (edge_sums.ndim != 2 or edge_sums.shape[0] != edge_sums.shape[1]
            or edge_counts.shape != edge_sums.shape or forbidden.shape != edge_sums.shape):
        raise ValueError('edge sums, counts and constraints must be matching square matrices')
    if (not 0 < threshold < 1 or not torch.isfinite(edge_sums).all()
            or not torch.isfinite(edge_counts).all() or (edge_sums < 0).any()
            or (edge_counts < 0).any()):
        raise ValueError('edge evidence must be finite and nonnegative, with a valid merge threshold')
    sums, counts, forbidden = edge_sums.clone(), edge_counts.clone(), forbidden.clone()
    count = len(sums)
    active = torch.ones(count, device=sums.device, dtype=torch.bool)
    parent = torch.arange(count, device=sums.device)
    while int(active.sum()) > 1:
        affinity = sums / (counts + 1e-6)
        affinity.fill_diagonal_(float('-inf'))
        affinity.masked_fill_(forbidden | ~active[:, None] | ~active[None, :], float('-inf'))
        i, j = divmod(int(affinity.argmax()), count)
        if float(affinity[i, j]) <= threshold:
            break
        merged_sums, merged_counts = sums[i] + sums[j], counts[i] + counts[j]
        sums[i, :], sums[:, i] = merged_sums, merged_sums
        counts[i, :], counts[:, i] = merged_counts, merged_counts
        blocked = forbidden[i] | forbidden[j]
        forbidden[i, :], forbidden[:, i] = blocked, blocked
        active[j] = False
        parent[parent == j] = i
    return torch.unique(parent, return_inverse=True)[1]


def cluster_graph(adjacency, labels, confidence, evidence, merge_threshold=.5, budget=128,
                  non_neighbor_weight=1.):
    """Quality spectral selection with mandatory frame constraints and four starts."""
    if not 0 < merge_threshold < 1 or budget < 1:
        raise ValueError('invalid merge threshold or cluster budget')
    if (isinstance(non_neighbor_weight, bool) or not isinstance(non_neighbor_weight, (int, float))
            or not math.isfinite(non_neighbor_weight) or not 0 <= non_neighbor_weight <= 1):
        raise ValueError('clustering.non_neighbor_weight must lie between zero and one')
    if len(adjacency) == 0:
        return labels.clone(), labels.clone(), confidence.clone()
    if evidence is None:
        raise ValueError('clustering requires frame evidence')
    budget = min(budget, len(adjacency))
    try:
        fine = quality_spectral_labels(adjacency, evidence, budget)
    except torch.linalg.LinAlgError:
        warnings.warn('Spectral decomposition failed; using constrained average linkage.', RuntimeWarning)
        fine = torch.arange(len(adjacency), device=adjacency.device)
    fine = lib_cluster.split_disconnected_clusters(adjacency, fine)
    fine = split_conflicting_clusters(fine, evidence.cannot_link, adjacency)
    count = int(fine.max()) + 1
    sizes = torch.bincount(fine, minlength=count)
    membership = F.one_hot(fine, count).float()
    forbidden = membership.T @ evidence.cannot_link.float() @ membership > 0
    if non_neighbor_weight == 1:
        # Preserve the legacy arithmetic and rounding for the baseline.
        affinity = lib_cluster.compute_cluster_edge_weights_optimized(adjacency, fine, count)
        merged = constrained_average_linkage(affinity, sizes, forbidden, merge_threshold)[fine]
    else:
        spatial = evidence.spatial_support
        if (spatial is None or spatial.shape != adjacency.shape or spatial.dtype != torch.bool
                or spatial.device != adjacency.device or not torch.equal(spatial, spatial.T)
                or ((adjacency > 0) & ~spatial).any()):
            raise ValueError('weighted merging requires matching spatial support for every positive edge')
        edge_sums = membership.T @ adjacency @ membership
        neighbor_counts = membership.T @ spatial.float() @ membership
        mass = membership.sum(0)
        edge_counts = (non_neighbor_weight * mass[:, None] * mass[None, :]
                       + (1 - non_neighbor_weight) * neighbor_counts)
        merged = constrained_edge_linkage(edge_sums, edge_counts, forbidden, merge_threshold)[fine]
    confidence = lib_cluster.compute_cluster_confidence_scores(merged, confidence)
    return fine[labels], merged[labels], confidence
