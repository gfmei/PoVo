import numpy as np
import pytest
import torch
from PIL import Image

from libs.instance_recognition import (TEXT_TEMPLATES, class_text_embeddings, classify_instance_views, expanded_box,
                                       focus_instance, instance_class_confidence, instance_crop,
                                       parse_rerank_response, rerank_candidates, select_instance_views)


def test_crop_preserves_edges_and_clips_expansion():
    image = Image.new('RGB', (40, 20), 'red')
    crop = instance_crop(image, (0, 0, 40, 10))
    pixels = np.asarray(crop)
    assert crop.size == (40, 40)
    assert (pixels[:, 0] == (255, 0, 0)).all(1).sum() == 10
    assert expanded_box((0, 0, 10, 10), image.size, 2) == (0, 0, 15, 15)


def test_target_focus_aligns_mask_and_preserves_target_colors():
    image = Image.new('RGB', (4, 2), (255, 0, 0))
    foreground = Image.fromarray(np.array([[255, 0]], dtype=np.uint8))
    focused = np.asarray(focus_instance(image, foreground))
    np.testing.assert_array_equal(focused[:, :2], np.asarray(image)[:, :2])
    assert (focused[:, 2:, 0] < 255).all()
    assert (focused[:, 2:, 1:] > 0).all()
    assert focus_instance(image, Image.new('L', (2, 1))).tobytes() == image.tobytes()
    with pytest.raises(ValueError, match='context'):
        focus_instance(image, foreground, context=1.1)


def test_text_ensemble_normalizes_prompts_and_preserves_class_order_across_batches():
    # A large norm on one wording must not outweigh the other five prompts.
    names = [f'object{i}' for i in range(44)]
    tokens = {template.format(name): [index % 2, prompt]
              for index, name in enumerate(names) for prompt, template in enumerate(TEXT_TEMPLATES)}
    class Encoder:
        logit_scale = torch.tensor(0.)
        def encode_text(self, batch):
            result = torch.zeros(len(batch), 2)
            directions = torch.where(batch[:, 1] == 0, batch[:, 0], 1 - batch[:, 0])
            result[torch.arange(len(batch)), directions] = torch.where(batch[:, 1] == 0, 10., 1.)
            return result
    embeddings = class_text_embeddings(Encoder(), names, lambda texts: torch.tensor([tokens[t] for t in texts]))
    expected = torch.tensor([[1., 5.], [5., 1.]]).repeat(22, 1) / np.sqrt(26)
    torch.testing.assert_close(embeddings, expected)
    with pytest.raises(ValueError, match='at least one class'):
        class_text_embeddings(Encoder(), [], lambda texts: torch.empty(0, 2))


def test_view_selection_prefers_new_camera_and_is_deterministic():
    views = [{'frame': 1, 'quality': 1., 'camera': [0, 0]},
             {'frame': 2, 'quality': .99, 'camera': [0, 0]},
             {'frame': 3, 'quality': .9, 'camera': [10, 0]}]
    assert [v['frame'] for v in select_instance_views(views, 2)] == [1, 3]
    assert select_instance_views(views[::-1], 2) == select_instance_views(views, 2)


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable'))])
def test_matched_embeddings_pool_weights_and_preserve_unknown(device):
    embeddings = torch.tensor([[1., 0.], [0., 1.]], device=device)
    owners = torch.tensor([0, 0], device=device)
    weights = torch.tensor([3., 1.], device=device)
    probabilities, similarities, supported = classify_instance_views(embeddings, owners, weights, embeddings, 10, 2)
    assert probabilities[0].argmax().item() == 0
    assert supported.tolist() == [True, False]
    assert probabilities[1].sum().item() == 0
    torch.testing.assert_close(similarities[0], torch.tensor([3., 1.], device=device) / np.sqrt(10))
    with pytest.raises(ValueError, match='dimensions'):
        classify_instance_views(embeddings, owners, weights, torch.ones(2, 3, device=device), 10, 2)


def test_reranking_uses_only_ambiguous_supported_instances():
    probabilities = torch.tensor([[.98, .02], [.51, .49], [.1, .9]])
    similarities = torch.tensor([[.7, .1], [.41, .4], [.2, .7]])
    indices, candidates = rerank_candidates(probabilities, similarities, torch.tensor([True, True, False]))
    assert indices == [1]
    assert candidates[1].tolist() == [0, 1]


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable'))])
def test_confidence_tracks_reranked_class_and_unknown_labels(device):
    probabilities = torch.tensor([[.9, .1], [.7, .3], [.8, .2]], device=device)
    classes = torch.tensor([1, 0, -1], device=device)
    objectness = torch.tensor([.5, .5, 1.], device=device)
    confidence = instance_class_confidence(probabilities, classes, objectness)
    torch.testing.assert_close(confidence, torch.tensor([.05, .35, 0.], device=device))


@pytest.mark.parametrize('response, expected', [('{"label":"Office Chair"}', 'office chair'),
                                              ('```json\n{"label":"chair"}\n```', 'chair'),
                                              ('{"label":"unknown"}', None),
                                              ('{"label":"table"}', None),
                                              ('chair', None), ('{"label":4}', None)])
def test_reranker_rejects_invalid_or_out_of_vocabulary_labels(response, expected):
    assert parse_rerank_response(response, ['chair', 'office chair']) == expected
