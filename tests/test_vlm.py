"""Verify the retained Qwen reranker and its cache validation."""

import json
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from models.vlm import VisionLanguageAssistant, generate_image_response
from tools.recognize_instances import rerank, validate_rerank_cache


@pytest.mark.parametrize('model_type', ['qwen3_5', 'llava_next'])
def test_generation_excludes_prompt_and_uses_model_specific_template(model_type):
    class Inputs(dict):
        def to(self, device):
            assert device == 'cpu'
            return self

    class Processor:
        def apply_chat_template(self, messages, **kwargs):
            if model_type == 'qwen3_5':
                assert kwargs['enable_thinking'] is False
            else:
                assert 'enable_thinking' not in kwargs
            assert kwargs['tokenize'] and kwargs['add_generation_prompt']
            assert all(item['image'].mode == 'RGB' for item in messages[0]['content'][:-1])
            return Inputs(input_ids=torch.tensor([[100, 101, 102]]))

        def batch_decode(self, ids, **kwargs):
            assert ids.tolist() == [[201, 202]]
            return ['{"label":"chair"}']

    class Model:
        device = 'cpu'
        config = SimpleNamespace(model_type=model_type)

        def generate(self, **kwargs):
            assert kwargs['do_sample'] is False
            assert kwargs['max_new_tokens'] == 64
            assert kwargs['repetition_penalty'] == 1.05
            assert kwargs['no_repeat_ngram_size'] == 6
            return torch.tensor([[100, 101, 102, 201, 202]])

    images = [Image.new('L', (4, 4)), Image.new('RGBA', (5, 5))]
    assert generate_image_response(images, 'Choose chair or table', Model(), Processor()) == '{"label":"chair"}'


@pytest.mark.parametrize('images, tokens', [([], 64), ([Image.new('RGB', (4, 4))], 0)])
def test_generation_rejects_empty_inputs(images, tokens):
    with pytest.raises(ValueError, match='At least one image'):
        generate_image_response(images, 'Choose a label', None, None, max_new_tokens=tokens)


def test_reranking_loads_once_and_rejects_changed_crops(tmp_path, monkeypatch):
    from tools import recognize_instances as workflow

    calls = []
    class Assistant:
        def generate(self, images, prompt, **kwargs):
            assert len(images) == 2
            assert all(image.mode == 'RGB' for image in images)
            assert 'chair' in prompt and kwargs['max_new_tokens'] == 64
            calls.append('generate')
            return '{"label":"chair"}'

    def load(*args, **kwargs):
        calls.append('load')
        return Assistant()

    monkeypatch.setattr(VisionLanguageAssistant, 'from_config', load)
    monkeypatch.setattr(workflow, 'rerank_candidates', lambda *args:
                        ([0], torch.arange(5).view(1, 5)))
    config = tmp_path / 'config.yaml'
    config.write_text('foundation_model:\n  reranker_model: Qwen/Qwen3.5-9B\n')
    args = SimpleNamespace(output_root=tmp_path, config=config, vlm_cache_dir=None,
                           local_files_only=True)
    scenes = ['first', 'second']
    for scene in scenes:
        root = tmp_path / scene
        (root / 'rgb').mkdir(parents=True)
        views = [[{'frame': index, 'native_box': [0, 0, 8, 8]} for index in [1, 2]]]
        (root / 'views.json').write_text(json.dumps(views))
        for index in [1, 2]:
            Image.new('RGB', (8, 8), 'red').save(root / 'rgb' / f'{index}.jpg')
        torch.save({'probabilities': torch.ones(1, 5) / 5, 'similarities': torch.zeros(1, 5),
                    'supported': torch.tensor([True]), 'mask_sha256': scene}, root / 'l336.pth')
    rerank(args, scenes)
    assert calls == ['load', 'generate', 'generate']
    rerank(args, scenes)
    assert calls == ['load', 'generate', 'generate']
    Image.new('RGB', (8, 8), 'blue').save(tmp_path / 'first/rgb/1.jpg')
    with pytest.raises(ValueError, match='different settings or inputs'):
        rerank(args, scenes)


def test_reranking_rejects_legacy_or_changed_checkpoints():
    metadata = {'model': 'Qwen/Qwen3.5-9B', 'revision': 'pinned'}
    cached = {'metadata': metadata, 'mask_sha256': 'same'}
    validate_rerank_cache(cached, metadata, 'same')
    for altered in [{'model': 'Qwen/Qwen2.5-VL-3B-Instruct'}, dict(metadata, revision='other')]:
        with pytest.raises(ValueError, match='different settings or inputs'):
            validate_rerank_cache(cached, altered, 'same')
    with pytest.raises(ValueError, match='different settings or inputs'):
        validate_rerank_cache({'mask_sha256': 'same'}, metadata, 'same')
