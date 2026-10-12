"""Instance crop selection and matched image/text recognition utilities."""

import json
import re

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps


TEXT_TEMPLATES = ('a {}', 'a photo of a {}', 'a photo of the {}',
                  'a close-up photo of a {}', 'a cropped photo of a {}', 'a {} in a room')


@torch.inference_mode()
def class_text_embeddings(model, class_names, tokenize):
    """Average normalized prompt embeddings in the matched CLIP text space."""
    device = model.logit_scale.device
    texts = [template.format(name) for name in class_names for template in TEXT_TEMPLATES]
    if not texts:
        raise ValueError('recognition requires at least one class name')
    encoded = torch.cat([F.normalize(model.encode_text(tokenize(texts[start:start + 256]).to(device)).float(), dim=-1)
                         for start in range(0, len(texts), 256)])
    return F.normalize(encoded.reshape(len(class_names), len(TEXT_TEMPLATES), -1).mean(1), dim=-1)


def focus_instance(image, foreground, context=.25):
    """Keep target colors while dimming distractors, retaining some context."""
    if not 0 <= context <= 1:
        raise ValueError('retained context must lie between zero and one')
    image = image.convert('RGB')
    foreground = foreground.convert('L').resize(image.size, Image.Resampling.NEAREST)
    if foreground.getbbox() is None:
        return image
    background = Image.blend(Image.new('RGB', image.size, (127, 127, 127)), image, context)
    return Image.composite(image, background, foreground)


def expanded_box(box, image_size, expansion=1.):
    """Expand a pixel box around its center, retaining valid image coordinates."""
    if expansion < 1:
        raise ValueError('crop expansion must be at least one')
    left, top, right, bottom = box
    if right <= left or bottom <= top:
        raise ValueError('crop box must have positive area')
    width, height = image_size
    cx, cy = (left + right) / 2, (top + bottom) / 2
    half_width, half_height = (right - left) * expansion / 2, (bottom - top) * expansion / 2
    return (max(0, int(np.floor(cx - half_width))), max(0, int(np.floor(cy - half_height))),
            min(width, int(np.ceil(cx + half_width))), min(height, int(np.ceil(cy + half_height))))


def instance_crop(image, box, expansion=1.):
    """Pad rectangular crops so CLIP's center crop cannot discard object edges."""
    crop = image.crop(expanded_box(box, image.size, expansion))
    side = max(crop.size)
    return ImageOps.pad(crop, (side, side), color=(127, 127, 127))


def select_instance_views(views, count=3):
    """Select clear views while rewarding distinct camera positions/directions."""
    if count < 1:
        raise ValueError('at least one view is required')
    remaining = sorted(views, key=lambda view: (-view['quality'], view['frame']))
    selected = []
    while remaining and len(selected) < count:
        def score(view):
            if not selected:
                return view['quality']
            camera = np.asarray(view['camera'])
            distance = min(float(((camera - np.asarray(old['camera'])) ** 2).sum()) for old in selected)
            return view['quality'] * (.5 + .5 * (1 - np.exp(-distance)))
        index = max(range(len(remaining)), key=lambda i: score(remaining[i]))
        selected.append(remaining.pop(index))
    return selected


def classify_instance_views(embeddings, owners, weights, text_embeddings, logit_scale, count, return_features=False):
    """Pool normalized crop embeddings, then compare with matched text features."""
    text = F.normalize(text_embeddings.float(), dim=-1)
    if embeddings.shape[-1] != text.shape[-1]:
        raise ValueError('image and text embedding dimensions must match')
    if len(embeddings) != len(owners) or len(embeddings) != len(weights):
        raise ValueError('each crop needs an owner and a weight')
    features = text.new_zeros((count, text.shape[-1]))
    mass = text.new_zeros(count)
    if len(embeddings):
        embeddings = F.normalize(embeddings.float(), dim=-1)
        valid = torch.isfinite(embeddings).all(-1) & (embeddings.norm(dim=-1) > 0) & (weights > 0)
        features.index_add_(0, owners[valid], embeddings[valid] * weights[valid, None])
        mass.index_add_(0, owners[valid], weights[valid])
    features = F.normalize(features, dim=-1)
    similarities = features @ text.T
    probabilities = (float(logit_scale) * similarities).softmax(-1)
    probabilities[mass == 0] = 0
    result = probabilities, similarities, mass > 0
    return (*result, features) if return_features else result


def parse_rerank_response(response, candidates):
    """Accept a canonical candidate label or an explicit unknown response."""
    match = re.search(r'\{[^{}]*\}', response)
    if match is None:
        return None
    try:
        label = json.loads(match.group()).get('label')
    except (ValueError, AttributeError):
        return None
    if not isinstance(label, str):
        return None
    normalized = label.strip().casefold()
    return next((candidate for candidate in candidates if candidate.casefold() == normalized), None)


def instance_class_confidence(probabilities, classes, objectness):
    """Score the selected class, including labels changed by a reranker."""
    valid = (classes >= 0) & (classes < probabilities.shape[1])
    chosen = probabilities.gather(1, classes.clamp(0, probabilities.shape[1] - 1)[:, None]).squeeze(1)
    return (chosen * objectness).masked_fill(~valid, 0)


def rerank_candidates(probabilities, similarities, supported, max_probability=.55, margin=.015, top_k=5):
    """Select ambiguous predictions using model evidence alone, without labels."""
    top_values, top_ids = probabilities.topk(min(top_k, probabilities.shape[-1]), dim=-1)
    if probabilities.shape[-1] < 2:
        return [], top_ids
    cosine_margin = similarities.gather(1, top_ids[:, :1]).squeeze(1) - similarities.gather(1, top_ids[:, 1:2]).squeeze(1)
    ambiguous = supported & ((top_values[:, 0] < max_probability) | (cosine_margin < margin))
    return torch.where(ambiguous)[0].tolist(), top_ids
