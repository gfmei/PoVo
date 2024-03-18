import re
from io import BytesIO

import requests
import torch
from PIL import Image

from libs.lib_mask import draw_k2_boxes
from llava.constants import (DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_PLACEHOLDER,
                             IMAGE_TOKEN_INDEX)
from llava.conversation import conv_templates
from llava.mm_utils import get_anyres_image_grid_shape, get_model_name_from_path, process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.model.llava_arch import unpad_image
from llava.utils import disable_torch_init
from libs.lib_mask import get_boxed_images

import spacy

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")


def extract_entities(sentence):
    # Phrase to remove
    phrase_to_remove = "the red bounding box in the image contains"

    # Remove the phrase from the sentence
    modified_sentence = sentence.replace(phrase_to_remove, "").strip()

    # Perform entity extraction on the modified sentence
    doc = nlp(modified_sentence)

    # Extract entities and return them as a list
    entities = [entity.text for entity in doc.ents]

    return entities


def produce_conv(model, qs, conv_mode):
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    return conv


def load_llava_model(model_path="liuhaotian/llava-v1.6-mistral-7b"):
    """Load Llava"""
    disable_torch_init()

    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, None, model_name
    )
    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    return tokenizer, model, image_processor, conv_mode


def get_image_category_from_llava(images, tokenizer, model, image_processor, prompt, image_size=None):
    # Model
    if image_size is None:
        image_size = [320, 240]
    images = [x.resize(image_size) for x in images]
    image_sizes = [x.size for x in images]  # a list of image size as [w, h]
    images_tensor = process_images(
        images,
        image_processor,
        model.config
    ).to(model.device, dtype=torch.float16)
    # prompt = conv.get_prompt()
    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )
    input_id_list = torch.cat([input_ids for _ in range(len(images_tensor))], dim=0)
    with torch.inference_mode():
        output_ids = model.generate(
            input_id_list,
            images=images_tensor,
            image_sizes=image_sizes,
            do_sample=True,
            temperature=0.2,
            top_p=None,
            num_beams=1,
            max_new_tokens=64,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id
        )
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    labels = []
    for output in outputs:
        if 'no objects' in output or 'too blurr' in output:
            labels.append(['wall'])
            continue
        if 'red bounding box' in output:
            new_names = extract_entities(output)
            if len(new_names) == 0:
                labels.append('floor')
            else:
                labels.extend(new_names)
            continue
        # Split the output into individual labels, convert each to lowercase and strip whitespace
        new_labels = [label.strip().lower().replace('.', '') for label in output.strip().split(',')]
        # Update the labels set with these new labels
        if '' in new_labels:
            new_labels.remove('')
        new_labels = [item for item in new_labels if
                      not (item.isalpha() and len(item) == 1) and not (item.isdigit() and len(item) == 1)]
        if new_labels is not None:
            labels.append(new_labels)
        else:
            labels.append('floor')
        # Now 'labels' contains all unique, cleaned labels from 'outputs'
    return labels


def generate_category_from_llava(image_files, tokenizer, model, image_processor, conv, image_size=None):
    # Model
    if image_size is None:
        image_size = [320, 240]
    # disable_torch_init()
    images = load_images(image_files, image_size)  # a list of images with shape [h, w, 3]
    # images, label_list = load_image_with_box(image_files[0], image_size, k=k)
    # print([label.sum() for label in label_list], len(label_list))
    image_sizes = [x.size for x in images]  # a list of image size as [w, h]
    images_tensor = process_images(
        images,
        image_processor,
        model.config
    ).to(model.device, dtype=torch.float16)

    prompt = conv.get_prompt()
    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )
    input_id_list = torch.cat([input_ids for _ in range(len(images_tensor))], dim=0)
    with torch.inference_mode():
        output_ids = model.generate(
            input_id_list,
            images=images_tensor,
            image_sizes=image_sizes,
            do_sample=True,
            temperature=0.2,
            top_p=None,
            num_beams=1,
            max_new_tokens=64,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id
        )
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    labels = set()
    for output in outputs:
        if 'no objects' in output or 'red box' in output:
            labels.update(['wall'])
            continue
        # Split the output into individual labels, convert each to lowercase and strip whitespace
        new_labels = [label.strip().lower().replace('.', '') for label in output.strip().split(',')]
        # Update the labels set with these new labels
        if '' in new_labels:
            new_labels.remove('')
        if new_labels is not None:
            labels.update(new_labels)
        # Now 'labels' contains all unique, cleaned labels from 'outputs'
    return labels


def image_parser(args):
    out = args.image_file.split(args.sep)
    return out


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files, image_size=None, box=False):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        if image_size is not None:
            image = image.resize(image_size)
        if box > 0:
            image = draw_k2_boxes(image, box, gap=2)

        out.append(image)
    return out


def load_image_with_box(root, image_size=None, k=-1):
    image = load_image(root)
    label_list = None
    if image_size is not None:
        image = image.resize(image_size)
    if k > 0:
        images, label_list = get_boxed_images(image, k)
    else:
        images = [image]
    return images, label_list


def decom_llava_data(images, model, image_sizes=None):
    if type(images) is list or images.ndim == 5:
        if type(images) is list:
            images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
        concat_images = torch.cat([image for image in images], dim=0)
        image_features = model.encode_images(concat_images)
        split_sizes = [image.shape[0] for image in images]
        image_features = torch.split(image_features, split_sizes, dim=0)
        mm_patch_merge_type = getattr(model.config, 'mm_patch_merge_type', 'flat')
        image_aspect_ratio = getattr(model.config, 'image_aspect_ratio', 'square')

        pad_image_features = []
        if mm_patch_merge_type == 'flat':
            image_features = [x.flatten(0, 1) for x in image_features]
        elif mm_patch_merge_type.startswith('spatial'):
            new_image_features = []
            for image_idx, image_feature in enumerate(image_features):
                if image_feature.shape[0] > 1:
                    base_image_feature = image_feature[0]
                    pad_image_feature = image_feature[1:]
                    height = width = model.get_vision_tower().num_patches_per_side
                    assert height * width == base_image_feature.shape[0]
                    if image_aspect_ratio == 'anyres':
                        num_patch_width, num_patch_height = get_anyres_image_grid_shape(
                            image_sizes[image_idx], model.config.image_grid_pinpoints,
                            model.get_vision_tower().config.image_size)
                        pad_image_feature = pad_image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                        # print('anyres image feature', image_feature.shape)
                    else:
                        raise NotImplementedError
                    if 'unpad' in mm_patch_merge_type:
                        # @Ryan  (num_patch_height, num_patch_width, height, width, dim) --> [4096, 18, 24]
                        # padding image size (dim, num_patch_height * height, num_patch_width * width)
                        pad_image_feature = pad_image_feature.permute(4, 0, 2, 1, 3).contiguous()
                        pad_image_feature = pad_image_feature.flatten(1, 2).flatten(2, 3)
                        pad_image_feature = unpad_image(pad_image_feature, image_sizes[image_idx])
                        # [4096, 18*24]
                        pad_image_feature = torch.cat((
                            pad_image_feature,
                            model.model.image_newline[:, None, None].expand(*pad_image_feature.shape[:-1], 1).to(
                                pad_image_feature.device)
                        ), dim=-1)  # [4096, 18, 25]
                        pad_image_feature = pad_image_feature.flatten(1, 2).transpose(0, 1).detach()
                    else:
                        image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                        image_feature = image_feature.flatten(0, 3)
                        pad_image_feature = image_feature
                    image_feature = base_image_feature
                else:
                    image_feature = image_feature[0]
                    if 'unpad' in mm_patch_merge_type:
                        image_feature = torch.cat((
                            image_feature,
                            model.model.image_newline[None]
                        ), dim=0)
                    pad_image_feature = image_feature
                new_image_features.append(image_feature.detach())
                pad_image_features.append(pad_image_feature)
            image_features = torch.stack(new_image_features, dim=0)
            pad_image_features = torch.stack(pad_image_features, dim=0)
        else:
            raise ValueError(f"Unexpected mm_patch_merge_type: {model.config.mm_patch_merge_type}")
    else:
        image_features = model.encode_images(images)
        pad_image_features = image_features

    return image_features, pad_image_features
