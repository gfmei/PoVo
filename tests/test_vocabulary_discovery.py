"""Check model-output boundaries without downloading checkpoints."""

from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from models.vocabulary import discover_image_categories, parse_object_names
from tools.discover_vocabulary import vocabulary_recall, validate_cached_image
from tools.evaluate_vocabulary import taxonomy_class_ids, image_hash
from models.povo import SAM2Mask
from libs.inference import score_instances


@pytest.mark.parametrize('answer, expected', [
    ('["chair", "office chair", "coffee table"]', ['chair', 'office chair', 'coffee table']),
    ('```json\n["The Chair.", "chair", false, null, 123, ""]\n```', ['chair']),
    ('<think>Maybe ["hidden object"]</think>["sink", "paper towel dispenser"]',
     ['sink', 'paper towel dispenser']),
    ('<think>Maybe ["hidden object"]', []),
    ('I see a chair and a desk.', []),
    ('["chair", "table"', ['chair', 'table']),
    ('["chair", "office ch', ['chair']),
    ('["chair"; "wall"]', []),
    ('["chair", undefined]', []),
    ('{"objects": "chair"}', []),
    ('["table", "<image>", "box; execute", "washing machine"]', ['table', 'washing machine']),
])
def test_object_names_are_strings_and_compounds_survive(answer, expected):
    assert parse_object_names(answer) == expected


def test_category_limit_bounds_detector_work():
    import json
    assert len(parse_object_names(json.dumps([f'object {index}' for index in range(40)]))) == 20


def test_generation_excludes_prompt_and_disables_thinking():
    calls = []

    class Inputs(dict):
        def to(self, device):
            assert device == 'cpu'
            return self

    class Processor:
        def apply_chat_template(self, messages, **kwargs):
            calls.append(messages)
            assert kwargs['enable_thinking'] is False
            assert kwargs['tokenize'] and kwargs['add_generation_prompt']
            assert messages[0]['content'][0]['image'].mode == 'RGB'
            return Inputs(input_ids=torch.tensor([[100, 101, 102]]))

        def batch_decode(self, ids, **kwargs):
            assert ids.tolist() == [[201, 202]]
            return ['["office chair", "desk"]']

    class Model:
        device = 'cpu'
        config = SimpleNamespace(model_type='qwen3_5')

        def generate(self, **kwargs):
            assert kwargs['do_sample'] is False
            assert kwargs['max_new_tokens'] == 256
            assert kwargs['repetition_penalty'] == 1.05
            assert kwargs['no_repeat_ngram_size'] == 6
            return torch.tensor([[100, 101, 102, 201, 202]])

    images = [Image.new('L', (4, 4)), Image.new('RGBA', (5, 5))]
    categories, responses = discover_image_categories(
        images, Model(), Processor(), return_responses=True)
    assert categories == [['office chair', 'desk'], ['office chair', 'desk']]
    assert len(responses) == len(calls) == 2


def test_recall_preserves_fine_grained_classes_and_deduplicates():
    counts = vocabulary_recall(['chair', 'chair', 'table'], ['office chair', 'table'])
    assert counts == {'matched': 1, 'eligible': 2, 'recall': .5}
    assert vocabulary_recall([], [])['recall'] is None


def test_unknown_and_background_names_do_not_become_foreground_predictions():
    from exp.scannet200_splits import INSTANCE_CAT_SCANNET_200 as names
    mapped = taxonomy_class_ids(['office chair', 'chair', 'floor', 'unlisted object'], [-1, 0, 1, 2, 3, 7])
    assert mapped.tolist() == [-1, names.index('office chair'), names.index('chair'), -1, -1, -1]


def test_cached_image_vocabulary_is_used_without_a_language_model():
    model = SimpleNamespace(image_vocabulary=lambda image: ['office chair', 'desk'])
    assert SAM2Mask.get_image_level_names(model, Image.new('RGB', (4, 4))) == ['office chair', 'desk']


def test_pixel_hash_changes_with_the_image_content():
    original = Image.new('RGB', (640, 480))
    altered = original.copy()
    altered.putpixel((20, 20), (255, 0, 0))
    assert image_hash(original) != image_hash(altered)
    assert image_hash(original) == image_hash(original.convert('RGBA'))


def test_discovery_resume_rejects_changed_pixels_and_labels():
    image = Image.new('RGB', (640, 480))
    record = {'rgb_sha256': image_hash(image), 'response': '["chair"]', 'labels': ['chair']}
    validate_cached_image(record, image)
    with pytest.raises(ValueError, match='RGB image differs'):
        validate_cached_image(record, Image.new('RGB', (640, 480), 'red'))
    with pytest.raises(ValueError, match='category labels differ'):
        validate_cached_image(dict(record, labels=['table']), image)


@pytest.mark.parametrize('count', [0, 2])
def test_no_vocabulary_returns_unknown_class_ids_including_empty_scenes(count):
    confidence = torch.ones(count)
    scores, semantic, classes = score_instances(
        torch.zeros((4, 2)), None, torch.zeros((count, 4)), confidence, return_class_ids=True)
    assert semantic is None and classes.dtype == torch.long
    assert classes.tolist() == [-1] * count
    torch.testing.assert_close(scores, confidence)
