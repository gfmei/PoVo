"""Small numerical and inference checks that do not download model weights."""

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from datasets.data_loader import ScanReader
from libs import instance_clustering
from libs.mapper import PointCloudToImageMapper
from libs.spatial_neighbors import superpoint_curve_adjacency
from models import sam2mask


def test_radius_graph_does_not_read_uninitialized_distances(monkeypatch):
    original_empty = torch.empty

    def poisoned_empty(*args, **kwargs):
        result = original_empty(*args, **kwargs)
        if result.is_floating_point():
            result.fill_(float('nan'))
        return result

    monkeypatch.setattr(torch, 'empty', poisoned_empty)
    points = torch.tensor([[0., 0., 1.], [.1, 0., 1.], [3., 0., 1.]])
    mapping = torch.tensor([[1, 0, 0, 1], [1, 0, 1, 1], [1, 0, 2, 1]])
    adjacency, confidence = sam2mask.construct_superpoint_adjacency_graph_radius(
        points, torch.arange(3), [torch.ones((1, 1, 3), dtype=torch.bool)],
        [mapping], [torch.tensor([.8])], min_points=1,
    )
    assert torch.isfinite(adjacency).all()
    torch.testing.assert_close(adjacency, adjacency.T)
    assert adjacency[0, 1] > 0
    assert adjacency[0, 2] == adjacency[1, 2] == 0
    torch.testing.assert_close(confidence, torch.full((3,), .8))


@pytest.mark.parametrize('min_points', [1, 2, 5])
def test_radius_graph_empty_masks_and_filtered_superpoints(min_points):
    points = torch.tensor([[0., 0., 1.], [.01, 0., 1.], [.1, 0., 1.]])
    adjacency, confidence = sam2mask.construct_superpoint_adjacency_graph_radius(
        points, torch.tensor([0, 0, 9]), [], [], [], min_points=min_points,
    )
    assert adjacency.shape == (2, 2)
    assert confidence.shape == (2,)
    assert not confidence.any()
    assert not adjacency.any()  # Spatial proximity without mask evidence cannot create instances.


@pytest.mark.parametrize('device', ['cpu', pytest.param(
    'cuda', marks=pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is unavailable'))])
@pytest.mark.parametrize('batch_size', [1, 3, 32])
@pytest.mark.parametrize('counts', [[4, 7, 3, 6, 5], [28, 37, 3, 41, 5]])
def test_curve_window_distances_match_scalar_reference(device, batch_size, counts):
    # Include uneven sizes and a window covering this whole small cloud.
    generator = torch.Generator().manual_seed(4)
    points = torch.randn((sum(counts), 3), generator=generator).to(device)
    labels = torch.repeat_interleave(torch.tensor([0, 4, 9, 15, 20]), torch.tensor(counts)).to(device)
    unique = torch.unique(labels)
    expected = torch.zeros((len(unique), len(unique)), device=device)
    for i, first in enumerate(unique):
        for j in range(i + 1, len(unique)):
            distance = torch.cdist(points[labels == first], points[labels == unique[j]]).min()
            if distance < .8:
                expected[i, j] = expected[j, i] = torch.exp(-distance)
    actual = superpoint_curve_adjacency(
        points, labels, radius=.8, window_size=len(points), chunk_size=batch_size * 10)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize('use_torch', [False, True])
def test_mapper_occlusion_and_empty_visibility(use_torch):
    mapper = PointCloudToImageMapper([3, 3], intrinsics=np.eye(3))
    # The near point precedes the far point: overwriting would select the far point.
    points = np.array([[0., 0., 1.], [0., 0., 1.3], [0., 0., -1.]], dtype=np.float32)
    if use_torch:
        mapping = mapper.compute_mapping_torch(torch.eye(4), torch.from_numpy(points)).numpy()
        hidden = mapper.compute_mapping_torch(torch.eye(4), torch.tensor([[100., 0., 1.]])).numpy()
    else:
        mapping = mapper.compute_mapping(np.eye(4), points)
        hidden = mapper.compute_mapping(np.eye(4), np.array([[100., 0., 1.]]))
    assert mapping[:, 2].tolist() == [1, 0, 0]
    assert hidden.tolist() == [[0, 0, 0]]


@pytest.mark.parametrize('use_torch', [False, True])
def test_mapper_missing_depth_and_intrinsic_override(use_torch):
    mapper = PointCloudToImageMapper([3, 3], intrinsics=np.eye(3))
    points = np.array([[0., 0., 1.]], dtype=np.float32)
    override = np.eye(3)
    override[0, 2] = 1
    if use_torch:
        project = lambda **kw: mapper.compute_mapping_torch(np.eye(4), torch.from_numpy(points), **kw).numpy()
    else:
        project = lambda **kw: mapper.compute_mapping(np.eye(4), points, **kw)
    assert project(intrinsic=override)[0, 1] == 1
    assert project()[0, 1] == 0
    assert not project(depth=np.zeros((3, 3), dtype=np.float32))[:, 2].any()


@pytest.mark.parametrize('raw_spp', [[0, 4, 4, 10], [4, 4, 10, 10], [0, 0, 0, 0]])
def test_scan_reader_compacts_superpoints_and_preserves_ids(tmp_path, raw_spp):
    scene_path = tmp_path / 'scannet200' / 'val' / 'scene.pth'
    scene_path.parent.mkdir(parents=True)
    data = {
        'coords': np.zeros((4, 3), dtype=np.float64),
        'spts': np.array(raw_spp),
        'instance_ids': np.array([0, 2049, 2051, -1]),
        'labels': np.array([0, 2, 3, -100]),
    }
    torch.save(data, scene_path)
    points, spp, instance_ids, labels = ScanReader(str(tmp_path), 'cpu').get_pcd('scene', 'val')
    _, expected = np.unique(raw_spp, return_inverse=True)
    assert spp.tolist() == expected.tolist()
    assert points.dtype == torch.float32
    assert instance_ids.dtype == labels.dtype == torch.long
    assert instance_ids.tolist() == data['instance_ids'].tolist()


def test_grounded_data_save_load_and_legacy_path(tmp_path):
    reader = ScanReader(str(tmp_path), 'cpu')
    data = {'masks': torch.ones((1, 2, 3)), 'img_feat': torch.ones((1, 2)), 'conf': torch.tensor([.8])}
    reader.save_mask('/images/color/10.jpg', 'val', 'scene', data)
    loaded = reader.get_grounded_data('/images/color/10.jpg', 'val', 'scene')
    for key in data:
        torch.testing.assert_close(loaded[key], data[key])
    canonical = tmp_path / 'masks' / 'val' / 'scene' / 'sam2' / '10.pth'
    legacy = tmp_path / 'mask_sam2' / 'val' / 'scene' / 'sam2' / '10.pth'
    legacy.parent.mkdir(parents=True)
    canonical.rename(legacy)
    assert reader.get_grounded_data('/images/color/10.jpg', 'val', 'scene')['masks'].shape == (1, 2, 3)


def test_scan_reader_uses_sibling_pose_and_depth_directories(tmp_path):
    import cv2

    scene_path = tmp_path / 'colorful_data' / 'color_scene'
    for folder in ['color', 'pose', 'depth']:
        (scene_path / folder).mkdir(parents=True)
    np.savetxt(scene_path / 'pose' / '0.txt', np.eye(4))
    cv2.imwrite(str(scene_path / 'depth' / '0.png'), np.full((2, 3), 1000, dtype=np.uint16))
    reader = ScanReader(str(tmp_path), 'cpu')
    image_path = str(scene_path / 'color' / '0.jpg')
    np.testing.assert_array_equal(reader.get_pose(image_path), np.eye(4))
    assert reader.get_depth(image_path).tolist() == [[1000] * 3] * 2


class TinyReader:
    def get_pcd(self, scene_id, split):
        return (
            torch.tensor([[0., 0., 1.], [.01, 0., 1.], [1., 0., 1.], [1.01, 0., 1.]]),
            torch.tensor([0, 0, 8, 8]), torch.tensor([2049, 2049, 2051, 2051]),
            torch.tensor([2, 2, 3, 3]),
        )

    def get_image_dirs(self, scene_id, split):
        return ['frame.jpg']

    def get_depth(self, image_path):
        return np.ones((2, 4), dtype=np.float32)

    def get_pose(self, image_path):
        return np.eye(4, dtype=np.float32)

    def get_intrinsic(self, scene_id):
        return np.eye(3)

    def get_image(self, image_path):
        return Image.new('RGB', (4, 2))

    def get_grounded_data(self, image_path, split, scene_id):
        masks = torch.zeros((2, 2, 4), dtype=torch.bool)
        masks[0, :, 0] = True
        masks[1, :, 1] = True
        return {'masks': masks, 'img_feat': torch.eye(2), 'conf': torch.tensor([.8, .6])}


def tiny_model(model_type, reader=None):
    model = model_type.__new__(model_type)
    nn.Module.__init__(model)
    model.device = 'cpu'
    model.clip_dim = 2
    model.img_dim = [4, 2]
    model.depth_scale = 1
    model.data_reader = reader or TinyReader()
    model.pointcloud_mapper = PointCloudToImageMapper(model.img_dim, intrinsics=np.eye(3))
    model.text_encoder = lambda names: None if not names else torch.ones((len(names), 2)).half()
    return model


@pytest.mark.parametrize('model_type', [sam2mask.SAM2Mask])
def test_cached_inference_returns_one_score_per_mask_and_integer_ground_truth(model_type):
    torch.manual_seed(0)
    model = tiny_model(model_type)
    masks, confidence, semantic, instances, labels = model('scene', class_names=['chair', 'table'])
    assert masks.shape[1] == 4
    assert confidence.shape == (masks.shape[0],)
    assert semantic.shape == (4,)
    assert torch.isfinite(confidence).all()
    assert (masks.sum(dim=0) == 1).all()
    assert instances.dtype == labels.dtype == torch.long
    assert instances.tolist() == [2049, 2049, 2051, 2051]


def test_spectral_clustering_uses_normalized_laplacian_and_row_embeddings(monkeypatch):
    adjacency = torch.tensor([[0., 1., 0.], [1., 0., 2.], [0., 2., 0.]])
    expected = torch.tensor([
        [1., -1 / np.sqrt(3), 0.],
        [-1 / np.sqrt(3), 1., -2 / np.sqrt(6)],
        [0., -2 / np.sqrt(6), 1.],
    ], dtype=torch.float32)
    original_eigh = torch.linalg.eigh
    matrices = []

    def check_laplacian(matrix):
        matrices.append(matrix)
        if len(matrices) == 1:
            torch.testing.assert_close(matrix, expected)
        return original_eigh(matrix)

    def check_embeddings(embeddings, n_clusters, **kwargs):
        torch.testing.assert_close(embeddings.norm(dim=-1), torch.ones((1, 3)))
        return torch.ones((1, 3, 1)), None, None

    evidence = instance_clustering.FrameEvidence(
        adjacency, torch.zeros_like(adjacency, dtype=torch.bool), adjacency,
        torch.zeros_like(adjacency), torch.ones(3))
    monkeypatch.setattr(torch.linalg, 'eigh', check_laplacian)
    monkeypatch.setattr(instance_clustering, 'kmeans', check_embeddings)
    labels = instance_clustering.quality_spectral_labels(adjacency, evidence)
    assert labels.tolist() == [0, 0, 0]
    assert len(matrices) == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is unavailable')
@pytest.mark.parametrize('model_type', [sam2mask.SAM2Mask])
def test_cached_inference_on_cuda(model_type):
    reader = TinyReader()
    original_get_pcd = reader.get_pcd
    reader.get_pcd = lambda *args: tuple(value.cuda() for value in original_get_pcd(*args))
    model = tiny_model(model_type, reader)
    model.device = 'cuda'
    model.text_encoder = lambda names: torch.ones((len(names), 2), device='cuda').half()
    masks, confidence, semantic, instances, labels = model('scene', class_names=['chair', 'table'])
    assert masks.is_cuda and confidence.is_cuda and semantic.is_cuda
    assert confidence.shape == (masks.shape[0],)
    assert torch.isfinite(confidence).all()
    assert instances.tolist() == [2049, 2049, 2051, 2051]


@pytest.mark.parametrize('model_type', [sam2mask.SAM2Mask])
@pytest.mark.parametrize('case', ['empty_masks', 'no_images', 'invisible', 'invalid_pose', 'empty_cloud'])
def test_inference_handles_empty_and_invisible_inputs(model_type, case):
    reader = TinyReader()
    if case == 'empty_masks':
        reader.get_grounded_data = lambda *args: {'masks': None, 'img_feat': None, 'conf': None}
    elif case == 'no_images':
        reader.get_image_dirs = lambda *args: []
    elif case == 'invisible':
        reader.get_depth = lambda *args: np.zeros((2, 4), dtype=np.float32)
    elif case == 'invalid_pose':
        reader.get_pose = lambda *args: np.full((4, 4), np.inf, dtype=np.float32)
    else:
        reader.get_pcd = lambda *args: (
            torch.empty((0, 3)), torch.empty(0, dtype=torch.long),
            torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long))
    model = tiny_model(model_type, reader)
    masks, confidence, semantic, instances, labels = model('scene', class_names=[])
    assert masks.shape == (0, len(instances))
    assert confidence.shape == (masks.shape[0],)
    assert not confidence.any()
    assert semantic is None
    assert instances.dtype == labels.dtype == torch.long


@pytest.mark.parametrize('model_type', [sam2mask.SAM2Mask])
@pytest.mark.parametrize('force_fallback', [False, True])
def test_inference_keeps_touching_objects_separate(model_type, force_fallback, monkeypatch):
    reader = TinyReader()
    original_get_pcd = reader.get_pcd
    def get_pcd(*args):
        _, spp, instances, labels = original_get_pcd(*args)
        points = torch.tensor([[0., 0., 1.], [.01, 0., 1.], [.05, 0., 1.], [.06, 0., 1.]])
        return points, spp, instances, labels
    reader.get_pcd = get_pcd
    reader.get_depth = lambda *args: np.ones((2, 8), dtype=np.float32)
    reader.get_intrinsic = lambda *args: np.diag([100., 1., 1.])
    masks = torch.zeros((2, 2, 8), dtype=torch.bool)
    masks[0, :, :2] = True
    masks[1, :, 5:7] = True
    reader.get_grounded_data = lambda *args: {
        'masks': masks, 'img_feat': torch.ones((2, 2)), 'conf': torch.tensor([.8, .6])}
    model = tiny_model(model_type, reader)
    model.img_dim = [8, 2]
    model.pointcloud_mapper = PointCloudToImageMapper(model.img_dim)
    if force_fallback:
        def failure(*args, **kwargs):
            raise torch.linalg.LinAlgError('forced eigendecomposition failure')
        monkeypatch.setattr(instance_clustering, 'quality_spectral_labels', failure)
        with pytest.warns(RuntimeWarning, match='constrained average linkage'):
            predictions, confidence, *_ = model('scene', class_names=[])
    else:
        predictions, confidence, *_ = model('scene', class_names=[])
    assert predictions.shape == (2, 4)
    assert {tuple(mask.tolist()) for mask in predictions} == {(1., 1., 0., 0.), (0., 0., 1., 1.)}
    torch.testing.assert_close(confidence.float().sort().values, torch.tensor([.6, .8]), atol=5e-4, rtol=0)
