"""Superpoint contacts from overlapping windows on point-level spatial curves.

Morton encoding uses the byte lookup approach from PointTransformerV3:
https://github.com/Pointcept/PointTransformerV3/tree/main/serialization
Its MIT notice is retained in serialization_LICENSE.txt. The lookup is
extended to 21 bits per coordinate without batch identifiers.

The Hilbert encoder vectorizes the transpose algorithm used by Gabriel
Altay's hilbertcurve package (https://github.com/galtay/hilbertcurve).
Its MIT notice is retained in hilbert_LICENSE.txt. No runtime dependency on
that package is needed.
"""

from functools import lru_cache
import math

import torch


DEFAULT_CURVE_ORDERS = ('morton', 'hilbert', 'morton_yzx', 'hilbert_yzx')
_AXES = {'xyz': (0, 1, 2), 'yzx': (1, 2, 0), 'zxy': (2, 0, 1)}


@lru_cache(maxsize=None)
def _morton_lut(device):
    """Cache each byte's interleaved axis bits once per tensor device."""
    byte = torch.arange(256, dtype=torch.long)
    spread = torch.zeros_like(byte)
    for bit in range(8):
        spread |= ((byte >> bit) & 1) << (3 * bit)
    return tuple(table.to(device) for table in (spread << 2, spread << 1, spread))


def _interleave(coordinates, bits):
    x_table, y_table, z_table = _morton_lut(coordinates.device)
    # The first byte always exists, including for an empty point tensor.
    x, y, z = coordinates.unbind(1)
    code = x_table[x & 255] | y_table[y & 255] | z_table[z & 255]
    for shift in range(8, bits, 8):
        chunk = (x_table[(x >> shift) & 255]
                 | y_table[(y >> shift) & 255]
                 | z_table[(z >> shift) & 255])
        code |= chunk << (3 * shift)
    return code


def encode_spatial_curve(coordinates, bits, curve):
    """Encode nonnegative integer XYZ coordinates into signed 64-bit keys."""
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError('Curve coordinates must have shape (N, 3).')
    if coordinates.dtype != torch.long or not 1 <= bits <= 21:
        raise ValueError('Curve coordinates must be int64 and bits must be between 1 and 21.')
    if coordinates.numel() and (bool((coordinates < 0).any()) or bool((coordinates >= 1 << bits).any())):
        raise ValueError('Coordinates exceed the selected curve bit depth.')
    kind, _, orientation = curve.partition('_')
    orientation = orientation or 'xyz'
    if kind not in ('morton', 'hilbert') or orientation not in _AXES:
        raise ValueError(f'Unsupported spatial curve: {curve}')
    axes = coordinates[:, _AXES[orientation]].clone()
    if kind == 'hilbert':
        # Undo the orientation changes at successive binary subdivisions.
        for bit in range(bits - 1, 0, -1):
            high_bit = 1 << bit
            lower_bits = high_bit - 1
            for axis in range(3):
                invert = (axes[:, axis] & high_bit) != 0
                exchange = (axes[:, 0] ^ axes[:, axis]) & lower_bits
                exchange = torch.where(invert, torch.zeros_like(exchange), exchange)
                axes[:, 0] ^= torch.where(invert, lower_bits, exchange)
                axes[:, axis] ^= exchange
        # Gray encoding and the final shared-axis correction.
        axes[:, 1] ^= axes[:, 0]
        axes[:, 2] ^= axes[:, 1]
        correction = torch.zeros(len(axes), dtype=torch.long, device=axes.device)
        for bit in range(bits - 1, 0, -1):
            correction ^= torch.where((axes[:, 2] & (1 << bit)) != 0, (1 << bit) - 1, 0)
        axes ^= correction[:, None]
    return _interleave(axes, bits)


def _refine_boundary_distances(points, inverse, candidates, radius):
    """Verify proposed superpoint pairs against all their boundary points.

    Rank windows can propose a pair from points away from its closest contact.
    Querying the entire superpoint avoids making that window-selected point
    pair, a center, or a small representative sample the final distance test.
    """
    import numpy as np
    from scipy.spatial import cKDTree

    count = candidates.shape[0]
    points_np = points.detach().cpu().double().numpy()
    labels_np = inverse.cpu().numpy()
    order = np.argsort(labels_np, kind='stable')
    counts = np.bincount(labels_np, minlength=count)
    boundaries = np.concatenate(([0], counts.cumsum()))
    groups = [order[boundaries[i]:boundaries[i + 1]] for i in range(count)]
    minima = np.stack([points_np[group].min(0) for group in groups])
    maxima = np.stack([points_np[group].max(0) for group in groups])
    box_gaps = np.maximum(minima[:, None] - maxima[None, :], minima[None, :] - maxima[:, None])
    box_gaps = np.maximum(box_gaps, 0)
    pairs = candidates.cpu().numpy()
    pairs &= (box_gaps * box_gaps).sum(2) < radius * radius
    pairs = np.triu(pairs, k=1)
    distances = np.full((count, count), np.inf, dtype=np.float32)
    for target in range(count):
        sources = np.flatnonzero(pairs[:, target])
        if not len(sources):
            continue
        source_points = np.concatenate([groups[source] for source in sources])
        tree = cKDTree(points_np[groups[target]])
        nearest, _ = tree.query(points_np[source_points], k=1, distance_upper_bound=radius)
        np.minimum.at(distances[:, target], labels_np[source_points], nearest)
    distances = np.minimum(distances, distances.T)
    return torch.as_tensor(distances, device=points.device)


def superpoint_curve_adjacency(points, superpoint_labels, radius=0.2, window_size=32,
                               resolution=0.01, chunk_size=8192,
                               curve_orders=DEFAULT_CURVE_ORDERS):
    """Find boundary contacts without superpoint centers or representative samples.

    Sort the actual points along each curve, collapse consecutive runs of the
    same superpoint label, and compare each run with the next window_size
    runs. Windows overlap, including across processing chunks. Different
    labels propose pairs. Verify each proposed pair against all
    its points, accepting it only when the minimum point-to-point distance is
    within radius. Weight the undirected edge by exp(-distance).

    Windows provide approximate candidate search: increasing their size or
    adding orientations can recover contacts missed by a finite run window.
    Distance validation prevents curve discontinuities from adding far edges.
    Rows follow sorted unique labels, including nonconsecutive input IDs.
    """
    if points.ndim != 2 or points.shape[1] != 3 or not points.is_floating_point():
        raise ValueError('points must be a floating tensor of shape (N, 3).')
    if superpoint_labels.ndim != 1 or len(superpoint_labels) != len(points):
        raise ValueError('superpoint_labels must contain one label per point.')
    if not math.isfinite(radius) or radius <= 0 or not math.isfinite(resolution) or resolution <= 0:
        raise ValueError('radius and curve resolution must be finite and positive.')
    if not isinstance(window_size, int) or window_size < 1 or not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError('curve window_size and chunk_size must be positive integers.')
    if isinstance(curve_orders, str) or not curve_orders or len(set(curve_orders)) != len(curve_orders):
        raise ValueError('curve_orders must be a nonempty sequence of distinct curve names.')
    for curve in curve_orders:
        kind, _, orientation = curve.partition('_')
        if kind not in ('morton', 'hilbert') or (orientation or 'xyz') not in _AXES:
            raise ValueError(f'Unsupported spatial curve: {curve}')
    if not bool(torch.isfinite(points).all()):
        raise ValueError('Point coordinates must be finite.')
    device = points.device
    labels, inverse = torch.unique(superpoint_labels.to(device).long(), return_inverse=True)
    count = len(labels)
    adjacency = torch.zeros((count, count), dtype=torch.float32, device=device)
    if len(points) < 2 or count < 2:
        return adjacency

    cells = torch.floor((points - points.amin(0)) / resolution).long()
    bits = max(1, int(cells.max()).bit_length())
    if bits > 21:
        raise ValueError('Scene extent exceeds 21-bit curve coordinates; increase curve resolution.')
    candidates_flat = torch.zeros(count * count, dtype=torch.bool, device=device)
    for curve in curve_orders:
        order = torch.argsort(encode_spatial_curve(cells, bits, curve), stable=True)
        ordered_labels = inverse[order]
        # A large superpoint must not fill the entire candidate window. Keep
        # every spatially separate occurrence, removing only consecutive repeats.
        run_starts = torch.ones(len(points), dtype=torch.bool, device=device)
        run_starts[1:] = ordered_labels[1:] != ordered_labels[:-1]
        ordered_labels = ordered_labels[run_starts]
        run_count = len(ordered_labels)
        offsets = torch.arange(1, min(window_size, run_count - 1) + 1, device=device)
        for start in range(0, run_count - 1, chunk_size):
            stop = min(run_count - 1, start + chunk_size)
            positions = torch.arange(start, stop, device=device)
            following = positions[:, None] + offsets[None, :]
            in_bounds = following < run_count
            following = following.clamp_max(run_count - 1)
            first_labels = ordered_labels[positions][:, None].expand_as(following)
            second_labels = ordered_labels[following]
            candidates = in_bounds & (first_labels != second_labels)
            first = first_labels[candidates]
            second = second_labels[candidates]
            candidates_flat[first * count + second] = True
    candidates = candidates_flat.view(count, count)
    candidates = candidates | candidates.T
    distances = _refine_boundary_distances(points, inverse, candidates, radius)
    finite = torch.isfinite(distances)
    adjacency[finite] = torch.exp(-distances[finite])
    adjacency.fill_diagonal_(0)
    return adjacency
