"""Reusable vision-language backend for Qwen recognition and LLaVA discovery."""

import torch

DEFAULT_MODEL = 'Qwen/Qwen3.5-9B'
DEFAULT_REVISION = 'c202236235762e1c871ad0ccb60c8ee5ba337b9a'
LLAVA_MODEL = 'llava-hf/llava-v1.6-mistral-7b-hf'
LLAVA_REVISION = '2424fdd47412fccc66d91719126b420e9fbd7065'


def load_vision_language_model(model_id=DEFAULT_MODEL, device='cuda', *,
                              revision=None, cache_dir=None, load_in_4bit=True,
                              local_files_only=False):
    """Load a pinned checkpoint once; NF4 permits inference on a 20 GB GPU."""
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

    if revision is None:
        revision = {DEFAULT_MODEL: DEFAULT_REVISION, LLAVA_MODEL: LLAVA_REVISION}.get(model_id)
    device = torch.device(device)
    if load_in_4bit and device.type != 'cuda':
        raise ValueError('4-bit vision-language inference requires CUDA; disable it for CPU inference.')
    dtype = (torch.bfloat16 if device.type == 'cuda' and torch.cuda.is_bf16_supported()
             else torch.float16 if device.type == 'cuda' else torch.float32)
    common = dict(revision=revision, cache_dir=cache_dir, local_files_only=local_files_only)
    processor_kwargs = dict(common)
    if model_id == DEFAULT_MODEL:
        processor_kwargs.update(min_pixels=65536, max_pixels=307200)
    processor = AutoProcessor.from_pretrained(model_id, **processor_kwargs)
    kwargs = dict(common, dtype=dtype, device_map={'': str(device)})
    if load_in_4bit:
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
    model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs).eval()
    if model.config.model_type == 'llava_next':
        # Older processor snapshots omit the image-token expansion settings.
        processor.patch_size = model.config.vision_config.patch_size
        processor.vision_feature_select_strategy = model.config.vision_feature_select_strategy
        processor.num_additional_image_tokens = 1  # CLIP adds a CLS token.
        processor.tokenizer.padding_side = 'left'
    return model, processor


@torch.inference_mode()
def generate_image_response(images, prompt, model, processor, *, max_new_tokens=64,
                            repetition_penalty=1.05, no_repeat_ngram_size=6):
    """Generate from one or more views, returning only the answer tokens."""
    if max_new_tokens < 1 or not images:
        raise ValueError('At least one image and a positive token limit are required.')
    messages = [{'role': 'user', 'content': [
        *[{'type': 'image', 'image': image.convert('RGB')} for image in images],
        {'type': 'text', 'text': prompt},
    ]}]
    template_kwargs = ({'enable_thinking': False}
                       if model.config.model_type == 'qwen3_5' else {})
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors='pt', **template_kwargs).to(model.device)
    generated = model.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=False,
        repetition_penalty=repetition_penalty, no_repeat_ngram_size=no_repeat_ngram_size)
    return processor.batch_decode(
        generated[:, inputs['input_ids'].shape[-1]:],
        skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


class VisionLanguageAssistant:
    """Reuse one loaded model across scenes in the recognition stage.

    There is no global model cache: releasing the assistant releases its weights.
    Requests run individually to bound memory for multi-view instance crops.
    """

    def __init__(self, model, processor):
        self.model = model
        self.processor = processor

    @classmethod
    def from_config(cls, foundation, device='cuda'):
        model, processor = load_vision_language_model(
            model_id=getattr(foundation, 'reranker_model', DEFAULT_MODEL),
            device=device,
            revision=getattr(foundation, 'reranker_revision', None),
            cache_dir=getattr(foundation, 'reranker_cache_dir', None),
            load_in_4bit=getattr(foundation, 'reranker_load_in_4bit', True),
            local_files_only=getattr(foundation, 'reranker_local_files_only', False))
        return cls(model, processor)

    def generate(self, images, prompt, **kwargs):
        return generate_image_response(images, prompt, self.model, self.processor, **kwargs)
