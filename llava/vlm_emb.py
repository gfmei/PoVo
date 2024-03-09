import os
import re
import sys

import numpy as np
import spacy
import torch
from torchvision.transforms import transforms

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append('../libs')
sys.path.append('../llava')

from llava.constants import (DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_PLACEHOLDER,
                             IMAGE_TOKEN_INDEX)
from llava.conversation import conv_templates
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.vlm_utils import load_images, load_image_with_box
from libs.vis_utils import visualize_feature_map

transformation = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        ),
    ]
)

nlp = spacy.load("en_core_web_sm")


def get_vlm_emb_from_diyclip(img_dir, model, img_size=None):
    '''Extract per-pixel OpenSeg features.'''
    # Model
    if img_size is None:
        img_size = [256, 256]
    disable_torch_init()

    images = load_images(img_dir, img_size)  # a list of images with shape [h, w, 3]
    images = torch.stack([torch.from_numpy(np.array(image)).permute(2, 0, 1) for image in images], dim=0)

    features = model(images)
    features = torch.nn.functional.normalize(features, dim=1)
    visualize_feature_map(features[0].flatten(start_dim=-2, end_dim=-1).transpose(0, 1).cpu().numpy(), 256, 256)

    return features


def get_vlm_emb_from_dinov2(img_dir, model, img_size=None):
    '''Extract per-pixel OpenSeg features.'''
    # Model
    if img_size is None:
        img_size = [336, 336]
    disable_torch_init()
    images = load_images(img_dir, img_size)  # a list of images with shape [h, w, 3]
    images = torch.stack([transformation(image) for image in images], dim=0)
    features = model(images)
    features = torch.nn.functional.normalize(features, dim=-1)

    visualize_feature_map(features[0].cpu().numpy(), 24, 24)
    bs, n, dim = features.shape
    features = features.reshape(bs, int(np.sqrt(n)), int(np.sqrt(n)), dim).permute(0, 3, 1, 2)

    return features


def get_vlm_emb_from_llava(image_files, model_path="liuhaotian/llava-v1.6-mistral-7b", patch_size=14, image_size=None):
    # Model
    if image_size is None:
        image_size = [320, 240]
    disable_torch_init()

    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, None, model_name
    )

    # images = load_images(image_files, image_size)  # a list of images with shape [h, w, 3]
    images = load_image_with_box(image_files, image_size, box=True)
    # images = torch.stack([transformation(image) for image in images], dim=0)
    # image_sizes = [x.size for x in images]  # a list of image size as [w, h]
    #
    # images_tensor = process_images(
    #     images,
    #     image_processor,
    #     model.config
    # ).to(model.device, dtype=torch.float16)
    images_tensor = image_processor.preprocess(images, return_tensors='pt')['pixel_values'].to(
        model.device, dtype=torch.float16)
    image_features = model.get_model().get_vision_tower()(images_tensor)
    print(image_features.shape, images_tensor.shape)
    visualize_feature_map(image_features[0].cpu().numpy(), 24, 24)
    # image_features, pad_image_features = decom_llava_data(images_tensor, model, image_sizes=image_sizes)
    #
    # if "llama-2" in model_name.lower():
    #     conv_mode = "llava_llama_2"
    # elif "mistral" in model_name.lower():
    #     conv_mode = "mistral_instruct"
    # elif "v1.6-34b" in model_name.lower():
    #     conv_mode = "chatml_direct"
    # elif "v1" in model_name.lower():
    #     conv_mode = "llava_v1"
    # elif "mpt" in model_name.lower():
    #     conv_mode = "mpt"
    # else:
    #     conv_mode = "llava_v0"
    # qs = 'List all objects in the images?'
    # image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    # if IMAGE_PLACEHOLDER in qs:
    #     if model.config.mm_use_im_start_end:
    #         qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
    #     else:
    #         qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    # else:
    #     if model.config.mm_use_im_start_end:
    #         qs = image_token_se + "\n" + qs
    #     else:
    #         qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
    # conv = conv_templates[conv_mode].copy()
    # conv.append_message(conv.roles[0], qs)
    # conv.append_message(conv.roles[1], None)
    # prompt = conv.get_prompt()
    # input_ids = (
    #     tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
    #     .unsqueeze(0)
    #     .cuda()
    # )
    # input_id_list = torch.cat([input_ids for _ in range(len(images_tensor))], dim=0)
    # print(input_id_list.shape)
    # with torch.inference_mode():
    #     output_ids = model.generate(
    #         input_id_list,
    #         images=images_tensor,
    #         image_sizes=image_sizes,
    #         do_sample=True,
    #         temperature=0.2,
    #         top_p=None,
    #         num_beams=1,
    #         max_new_tokens=512,
    #         use_cache=True,
    #         pad_token_id=tokenizer.eos_token_id
    #     )
    # outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    # print(outputs)
    # visualize_image(images[0])
    # image_feature = torch.nn.functional.normalize(image_features[0], dim=-1)
    # pad_image_feature = torch.nn.functional.normalize(pad_image_features[0], dim=-1)
    # # feature_cluster_visualization(image_feature[:orig_size].cpu().numpy(), n_clusters=8, w=24, h=24)
    # # feature_cluster_visualization(image_feature[orig_size:].cpu().numpy(), n_clusters=8, w=18, h=24)
    # visualize_feature_map(image_feature.cpu().numpy(), 24, 24)
    # visualize_feature_map(pad_image_feature.cpu().numpy(), 18, 25)
    # dim = image_features[0].shape[-1]
    # bs, _, _, h, w = images_tensor.shape
    dim = image_features.shape[-1]
    bs, _, h, w = images_tensor.shape
    image_features = image_features.reshape(bs, h // patch_size, w // patch_size, dim).permute(0, 3, 1, 2)
    return image_features, image_features


def get_category_from_llava(image_files, model_path="liuhaotian/llava-v1.6-mistral-7b", image_size=None, k=10):
    # Model
    if image_size is None:
        image_size = [320, 240]
    disable_torch_init()

    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, None, model_name
    )

    images = load_images(image_files, image_size)  # a list of images with shape [h, w, 3]
    # images, label_list = load_image_with_box(image_files[0], image_size, k=k)
    # print([label.sum() for label in label_list], len(label_list))
    image_sizes = [x.size for x in images]  # a list of image size as [w, h]
    images_tensor = process_images(
        images,
        image_processor,
        model.config
    ).to(model.device, dtype=torch.float16)

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
    # qs = 'What objects are within the red bounding box in the image? Please reply only contains the names of the objects.'
    qs = 'What objects are within the image? Please reply only contains the names of the objects.'
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
        # Split the output into individual labels, convert each to lowercase and strip whitespace
        new_labels = [label.strip().lower().replace('.', '') for label in output.strip().split(',')]
        print(new_labels)
        # Update the labels set with these new labels
        labels.update(new_labels)
        # Now 'labels' contains all unique, cleaned labels from 'outputs'
    print(labels)
    # doc = nlp(output)
    # nouns = [token.text for token in doc if token.pos_ == 'NOUN']
    # print(nouns)
    # visualize_image(images[0])
    # image_feature = torch.nn.functional.normalize(image_features[0], dim=-1)
    # pad_image_feature = torch.nn.functional.normalize(pad_image_features[0], dim=-1)
    # # feature_cluster_visualization(image_feature[:orig_size].cpu().numpy(), n_clusters=8, w=24, h=24)
    # # feature_cluster_visualization(image_feature[orig_size:].cpu().numpy(), n_clusters=8, w=18, h=24)
    # visualize_feature_map(image_feature.cpu().numpy(), 24, 24)
    # visualize_feature_map(pad_image_feature.cpu().numpy(), 18, 25)
    # dim = image_features[0].shape[-1]
    # bs, _, _, h, w = images_tensor.shape


if __name__ == '__main__':
    root = ['./test1.jpg']
    get_category_from_llava(root, k=6)
