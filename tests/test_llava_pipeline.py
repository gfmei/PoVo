"""Check the restored LLaVA path with synthetic inputs and no model downloads."""

from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch
import yaml

from libs.lib_data import BoundingBox, DetectionResult
from models import povo, vocabulary, vlm
from tests.test_pipeline import TinyReader, tiny_model


@pytest.mark.parametrize('model_id, revision, model_type', [
    (vlm.LLAVA_MODEL, vlm.LLAVA_REVISION, 'llava_next'),
    (vlm.DEFAULT_MODEL, vlm.DEFAULT_REVISION, 'qwen3_5'),
])
def test_loader_pins_checkpoint_and_configures_llava_image_tokens(
        monkeypatch, model_id, revision, model_type):
    processor = SimpleNamespace(tokenizer=SimpleNamespace(padding_side='right'))
    calls = []

    def load_processor(identifier, **kwargs):
        assert identifier == model_id
        assert kwargs['revision'] == revision and kwargs['local_files_only']
        calls.append('processor')
        return processor

    class Model:
        config = SimpleNamespace(model_type=model_type,
                                 vision_config=SimpleNamespace(patch_size=14),
                                 vision_feature_select_strategy='default')

        def eval(self):
            return self

    def load_model(identifier, **kwargs):
        assert identifier == model_id and kwargs['revision'] == revision
        assert kwargs['dtype'] == torch.float32 and kwargs['device_map'] == {'': 'cpu'}
        assert 'quantization_config' not in kwargs
        calls.append('model')
        return Model()

    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(
        AutoProcessor=SimpleNamespace(from_pretrained=load_processor),
        AutoModelForImageTextToText=SimpleNamespace(from_pretrained=load_model),
        BitsAndBytesConfig=SimpleNamespace))
    _, loaded_processor = vlm.load_vision_language_model(
        model_id, 'cpu', load_in_4bit=False, local_files_only=True)
    assert calls == ['processor', 'model']
    if model_type == 'llava_next':
        assert loaded_processor.patch_size == 14
        assert loaded_processor.num_additional_image_tokens == 1
        assert loaded_processor.vision_feature_select_strategy == 'default'
        assert loaded_processor.tokenizer.padding_side == 'left'


def test_live_vocabulary_uses_llava_configuration_independently_of_reranker(monkeypatch):
    calls = []

    def load(model_id, device, **kwargs):
        calls.append((model_id, device, kwargs))
        return 'llava-model', 'llava-processor'

    monkeypatch.setattr(vocabulary, 'load_vocabulary_model', load)
    foundation = SimpleNamespace(reranker_model=vlm.DEFAULT_MODEL,
                                 vocabulary_model=vlm.LLAVA_MODEL,
                                 vocabulary_revision=vlm.LLAVA_REVISION,
                                 vocabulary_load_in_4bit=False,
                                 vocabulary_local_files_only=True)
    model = SimpleNamespace(config=SimpleNamespace(foundation_model=foundation),
                            device='cpu', vlm_backend=None)
    povo.SAM2Mask._set_model(model)
    assert calls[0][0:2] == (vlm.LLAVA_MODEL, 'cpu')
    assert calls[0][2]['revision'] == vlm.LLAVA_REVISION
    assert calls[0][2]['local_files_only'] and not calls[0][2]['load_in_4bit']
    assert model.vlm_model == 'llava-model'
    povo.SAM2Mask._set_model(model)
    assert len(calls) == 1


@pytest.mark.parametrize('appearance_mode', ['clip', 'histogram_rgb'])
def test_original_pipeline_grounds_discovered_names_and_limits_inference_to_50_views(appearance_mode):
    class Reader(TinyReader):
        def get_image_dirs(self, scene, split):
            return [f'{index}.jpg' for index in range(60)]

    model = tiny_model(povo.SAM2Mask, Reader())
    config = yaml.safe_load((Path(__file__).resolve().parents[1] /
                             'configs/scannet200_llava.yaml').read_text())
    model.config = SimpleNamespace(data=SimpleNamespace(**config['data']),
                                    graph=SimpleNamespace(appearance_mode=appearance_mode))
    discovered, grounded = [], []

    def names(image):
        discovered.append(image)
        return ['office chair', 'coffee table']

    def detect(image, labels):
        grounded.append(labels)
        masks = model.data_reader.get_grounded_data('', '', '')['masks'].numpy()
        return [DetectionResult(.8, label, BoundingBox(index, 0, index + 1, 2), masks[index])
                for index, label in enumerate(labels)]

    model.image_vocabulary = names
    model.detect = detect
    model.segment = lambda image, detections, refinement: detections
    model.clip_preprocess = lambda image: torch.tensor([1., 0.])
    model.clip_adapter = SimpleNamespace(encode_image=lambda images: images)
    model.crop_and_extract_features = lambda image, detections: (
        povo.SAM2Mask.crop_and_extract_features(model, image, detections, thresh=1))
    masks, scores, semantic, instances, labels = model('synthetic')
    assert len(discovered) == len(grounded) == 50
    assert all(names == ['office chair', 'coffee table'] for names in grounded)
    assert model.discovered_class_names == ('office chair', 'coffee table')
    assert len(scores) == len(model.instance_class_ids) == len(masks) > 0
    assert torch.isfinite(scores).all() and torch.isfinite(model.instance_objectness).all()
    assert (masks.sum(0) <= 1).all()
    assert semantic.shape == (4,) and instances.dtype == labels.dtype == torch.long


def test_empty_scene_clears_vocabulary_and_prediction_state():
    class Reader(TinyReader):
        def get_pcd(self, scene, split):
            return torch.empty(0, 3), torch.empty(0), torch.empty(0), torch.empty(0)

    model = tiny_model(povo.SAM2Mask, Reader())
    model.discovered_class_names = ('stale label',)
    masks, scores, semantic, _, _ = model('empty')
    assert masks.shape == (0, 0) and not len(scores) and semantic is None
    assert model.discovered_class_names == () and not len(model.instance_class_ids)
