"""Image-only object vocabulary discovery with a local vision-language model."""

import json
import re

import torch

from models.vlm import (LLAVA_MODEL as DEFAULT_MODEL, LLAVA_REVISION as DEFAULT_REVISION,
                        generate_image_response, load_vision_language_model)


def load_vocabulary_model(model_id=DEFAULT_MODEL, device='cuda', **kwargs):
    """Load LLaVA for the original vocabulary-free PoVo pipeline."""
    return load_vision_language_model(model_id, device, **kwargs)


REPETITION_PENALTY = 1.05
NO_REPEAT_NGRAM_SIZE = 6
MAX_CATEGORIES = 20
PARSER_VERSION = 2
OBJECT_PROMPT = (
    'List the objects and background surfaces clearly visible in this indoor image. '
    'List at most 20 unique categories, each only once. '
    'Use short, specific, singular object category names, preserving compound names '
    'such as "office chair" or "coffee table". Avoid color, material, and room qualifiers. '
    'Include ordinary objects, not just '
    'unusual ones. Do not infer objects that are hidden or outside the image. '
    'Return only a compact JSON array of strings, without explanations.'
)


def parse_object_names(response):
    """Read complete JSON strings, retaining a valid prefix at a token cutoff."""
    if response.rfind('<think>') > response.rfind('</think>'):
        return []
    response = response.rsplit('</think>', 1)[-1]
    start = response.find('[')
    if start < 0:
        return []
    decoder = json.JSONDecoder()
    try:
        values, _ = decoder.raw_decode(response[start:])
    except json.JSONDecodeError:
        values = []
        remaining = response[start + 1:].lstrip()
        while remaining:
            try:
                value, end = decoder.raw_decode(remaining)
            except json.JSONDecodeError as error:
                if remaining.startswith('"') and error.msg.startswith('Unterminated string'):
                    break
                return []
            values.append(value)
            remaining = remaining[end:].lstrip()
            if not remaining:
                break
            if not remaining.startswith(','):
                return []
            remaining = remaining[1:].lstrip()
    if not isinstance(values, list):
        return []
    names = []
    for value in values:
        if not isinstance(value, str):
            continue
        name = re.sub(r'\s+', ' ', value).strip().lower().strip(' .')
        name = re.sub(r'^(?:a|an|the) ', '', name)
        if (name and len(name) <= 80 and len(name.split()) <= 8
                and re.fullmatch(r"[a-z0-9][a-z0-9 '/&()\-]*", name)
                and name not in names):
            names.append(name)
            if len(names) == MAX_CATEGORIES:
                break
    return names


@torch.inference_mode()
def discover_image_categories(images, model, processor, *, max_new_tokens=256,
                              return_responses=False):
    """Discover labels without a supplied class vocabulary or ground truth.

    Process images independently to bound vision-token and generation memory.
    Decode generated tokens only: prompt text must never enter the vocabulary.
    """
    if max_new_tokens < 1:
        raise ValueError('max_new_tokens must be positive.')
    categories, responses = [], []
    for image in images:
        response = generate_image_response(
            [image], OBJECT_PROMPT, model, processor, max_new_tokens=max_new_tokens,
            repetition_penalty=REPETITION_PENALTY, no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE)
        responses.append(response)
        categories.append(parse_object_names(response))
    return (categories, responses) if return_responses else categories
