import copy
import os
import re
import sys
from typing import List

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch import nn
from transformers import (CLIPVisionModelWithProjection, CLIPTokenizerFast, CLIPImageProcessor, CLIPModel,
                          CLIPTextModelWithProjection)

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append('../libs')
sys.path.append('../llava')

from llava.constants import DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_PLACEHOLDER
from llava.conversation import conv_templates
from llava.vlm_utils import get_image_category_from_llava, load_llava_model
from libs.lib_mask import assign_region_feature_to_image, gen_masked_imgs, get_boxed_images


class CLIPText(nn.Module):
    def __init__(self, config, device='auto', *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert device in ["auto", "cpu", "cuda"]
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
            # print("Using device: {}".format(device))
            self.device = torch.device(device)
        elif device == "cpu":
            self.device = torch.device("cpu")
            # print("Using device: {}".format(device))
        elif device == "cuda":
            self.device = torch.device("cuda")
            # print("Using device: {}".format(device))
        else:
            raise NotImplementedError
        self.config = config
        self.to_PIL = T.ToPILImage()
        self.to_t = T.ToTensor()

        self._set_model(config)

    @staticmethod
    def _create_prompt(text):
        return 'a photo of an ' + text if text.startswith('aeiou') else 'a photo of a ' + text

    def _set_model(self, config):
        self.tokenizer = CLIPTokenizerFast.from_pretrained(config.clip_model)
        self.text_model = CLIPTextModelWithProjection.from_pretrained(config.clip_model).eval()
        self.clip_model = CLIPModel.from_pretrained(config.clip_model)
        self.llava_tokenizer, self.model, self.image_processor, self.conv_mode = load_llava_model(
            model_path="liuhaotian/llava-v1.6-mistral-7b")
        qs = 'What objects are within the red bounding box in the image? ' \
             'Please reply only contains the names of the objects.'
        image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        if IMAGE_PLACEHOLDER in qs:
            if self.model.config.mm_use_im_start_end:
                qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
            else:
                qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
        else:
            if self.model.config.mm_use_im_start_end:
                qs = image_token_se + "\n" + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
        conv = conv_templates[copy.deepcopy(self.conv_mode)].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        conv = conv_templates[copy.deepcopy(self.conv_mode)].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        # prompt = conv.get_prompt()
        self.conv = conv

    def _embed_label(self, label: str) -> torch.Tensor:
        """
        Encode label name into a single vector
        """
        all_prompts = [self._create_prompt(label)]
        l_tokenized = self.tokenizer(all_prompts, return_tensors="pt", padding=True)
        out = self.text_model(**l_tokenized).text_embeds
        out = torch.mean(out, dim=0)
        return out

    def text_embedding(self, class_names: List[str]):
        aug_embeddings = torch.stack([self._embed_label(label) for label in class_names])
        # normalize vector
        # aug_embeddings = aug_embeddings / aug_embeddings.norm(p=2, dim=-1, keepdim=True)
        return aug_embeddings

    def get_image_level_names(self, img):
        # qs = 'List objects names that are within the image? Please reply only contains the names of the objects.'
        qs = 'What objects are within the image? Please reply only contains the names of the objects.'
        image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        if IMAGE_PLACEHOLDER in qs:
            if self.model.config.mm_use_im_start_end:
                qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
            else:
                qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
        else:
            if self.model.config.mm_use_im_start_end:
                qs = image_token_se + "\n" + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
        conv = conv_templates[copy.deepcopy(self.conv_mode)].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        cls_names = get_image_category_from_llava([img], self.llava_tokenizer, self.model, self.image_processor,
                                                  conv.get_prompt(), image_size=None)
        return cls_names[0]

    def forward(self, img, flag=True):
        img_list, box_list = get_boxed_images(img, n_segments=self.config.n_segments, flag=flag)
        # img_list = [img for img in img_list]
        with torch.no_grad():  # Disable gradient calculation for inference
            class_names = get_image_category_from_llava(img_list, self.tokenizer, self.model, self.image_processor,
                                                        self.conv.get_prompt(), image_size=None)
            # print('===============', class_names)
            text_embeds = [F.normalize(self.text_embedding(class_name).mean(0), dim=-1) for class_name in class_names]
        region_features = assign_region_feature_to_image(text_embeds, box_list, img.size)
        return region_features


class CLIPMeta(nn.Module):
    def __init__(self, config, device='auto', *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert device in ["auto", "cpu", "cuda"]
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
            # print("Using device: {}".format(device))
            self.device = torch.device(device)
        elif device == "cpu":
            self.device = torch.device("cpu")
            # print("Using device: {}".format(device))
        elif device == "cuda":
            self.device = torch.device("cuda")
            # print("Using device: {}".format(device))
        else:
            raise NotImplementedError
        self.config = config
        self.to_PIL = T.ToPILImage()
        self.to_t = T.ToTensor()

        self._set_model(config)

    @staticmethod
    def _create_prompt(text):
        return 'a photo of an ' + text if text.startswith('aeiou') else 'a photo of a ' + text

    def _set_model(self, config):
        self.image_model = CLIPVisionModelWithProjection.from_pretrained(config.clip_model)
        self.image_model = self.image_model.eval()
        self.tokenizer = CLIPTokenizerFast.from_pretrained(config.clip_model)
        self.text_model = CLIPTextModelWithProjection.from_pretrained(config.clip_model).eval()
        self.img_processor = CLIPImageProcessor.from_pretrained(config.clip_model)
        self.clip_model = CLIPModel.from_pretrained(config.clip_model)
        self.llava_tokenizer, self.model, self.image_processor, self.conv_mode = load_llava_model(
            model_path="liuhaotian/llava-v1.6-mistral-7b")

    def _embed_label(self, label: str) -> torch.Tensor:
        """
        Encode label name into a single vector
        """
        all_prompts = [self._create_prompt(label)]
        l_tokenized = self.tokenizer(all_prompts, return_tensors="pt", padding=True)
        out = self.text_model(**l_tokenized).text_embeds
        out = torch.mean(out, dim=0)
        return out

    def text_embedding(self, class_names: List[str]):
        aug_embeddings = torch.stack([self._embed_label(label) for label in class_names])
        # normalize vector
        aug_embeddings = aug_embeddings / aug_embeddings.norm(p=2, dim=-1, keepdim=True)
        return aug_embeddings

    def get_image_level_names(self, img):
        # qs = 'List objects names that are within the image? Please reply only contains the names of the objects.'
        qs = 'What objects are within the image? Please reply only contains the names of the objects.'
        image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        if IMAGE_PLACEHOLDER in qs:
            if self.model.config.mm_use_im_start_end:
                qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
            else:
                qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
        else:
            if self.model.config.mm_use_im_start_end:
                qs = image_token_se + "\n" + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
        conv = conv_templates[copy.deepcopy(self.conv_mode)].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        cls_names = get_image_category_from_llava([img], self.llava_tokenizer, self.model, self.image_processor,
                                                  conv.get_prompt(), image_size=None)
        return cls_names[0]

    def forward(self, img, flag=False):
        img_list, box_list = gen_masked_imgs(img, n_segments=self.config.n_segments, flag=flag)
        # img_list = [img for img in img_list]
        image_processed = self.img_processor(img_list, return_tensors="pt").to(self.image_model.device)
        with torch.no_grad():  # Disable gradient calculation for inference
            image_out = self.image_model(**image_processed)
            image_embeds = F.normalize(image_out.image_embeds, dim=-1)
        region_features = assign_region_feature_to_image(image_embeds, box_list, img.size)
        return region_features


class PatchCLIP(nn.Module):
    def __init__(self, config, device='auto', *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert device in ["auto", "cpu", "cuda"]
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
            # print("Using device: {}".format(device))
            self.device = torch.device(device)
        elif device == "cpu":
            self.device = torch.device("cpu")
            # print("Using device: {}".format(device))
        elif device == "cuda":
            self.device = torch.device("cuda")
            # print("Using device: {}".format(device))
        else:
            raise NotImplementedError
        self.config = config
        self.to_PIL = T.ToPILImage()
        self.to_t = T.ToTensor()
        self.patch_sizes = config.patch_sizes
        self.strides = config.strides
        self.interpolate_logits = config.interpolate_logits
        self.img_size = config.img_size
        self.align_corners = config.align_corners
        self.patch_weights = torch.tensor([config.patch_w1, config.patch_w2, config.patch_w3])
        self._set_model(config)

    def _set_model(self, config):
        self.image_model = CLIPVisionModelWithProjection.from_pretrained(config.clip_model)
        self.image_model = self.image_model.eval()
        self.tokenizer = CLIPTokenizerFast.from_pretrained(config.clip_model)
        self.text_model = CLIPTextModelWithProjection.from_pretrained(config.clip_model).eval()
        self.img_processor = CLIPImageProcessor.from_pretrained(config.clip_model)
        self.clip_model = CLIPModel.from_pretrained(config.clip_model)
        # extract class embeddings

        self.text_model = CLIPTextModelWithProjection.from_pretrained(config.clip_model).eval()
        self.llava_tokenizer, self.model, self.image_processor, self.conv_mode = load_llava_model(
            model_path="liuhaotian/llava-v1.6-mistral-7b")

    @staticmethod
    def _create_prompt(text):
        return 'a photo of an ' + text if text.startswith('aeiou') else 'a photo of a ' + text

    def clip_conv(self, img, patch_size=32, stride=2):
        B, _, h, w = img.shape
        patches = self._extract_patches(img, patch_size,
                                        stride)  # B, 3, npatch, hp, wp  (npatch = (hw // patch_size**2))
        patches = self._process_patches(patches)  # List[PIL.Image]  (B*npatch x (3, hp, wp))

        patches_sims = self.infer(patches)  # B*npatch, C
        num_classes = patches_sims.shape[-1]

        masks = self._group_patches(patches_sims, (B, num_classes, h, w))  # B, C, h, w

        return masks

    def forward(self, images, flag=False):
        images = images.resize((256, 256))
        images = self.to_t(images).unsqueeze(0)
        masks = []
        for ps in self.patch_sizes:
            for s in self.strides:
                masks.append(self.clip_conv(images, ps, s))
        mask = torch.mean(torch.stack(masks, dim=0) * self.patch_weights.view((-1,) + masks[0].dim() * (1,)), dim=0)
        mask = mask.permute(0, 2, 3, 1).squeeze(0)
        return mask

    def _embed_label(self, label: str) -> torch.Tensor:
        """
        Encode label name into a single vector
        """
        all_prompts = [self._create_prompt(label)]
        l_tokenized = self.tokenizer(all_prompts, return_tensors="pt", padding=True)
        out = self.text_model(**l_tokenized).text_embeds
        out = torch.mean(out, dim=0)
        return out

    def text_embedding(self, class_names: List[str]):
        aug_embeddings = torch.stack([self._embed_label(label) for label in class_names])
        # normalize vector
        aug_embeddings = aug_embeddings / aug_embeddings.norm(p=2, dim=-1, keepdim=True)
        return aug_embeddings

    def get_image_level_names(self, img):
        # qs = 'List objects names that are within the image? Please reply only contains the names of the objects.'
        qs = 'What objects are within the image? Please reply only contains the names of the objects.'
        image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        if IMAGE_PLACEHOLDER in qs:
            if self.model.config.mm_use_im_start_end:
                qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
            else:
                qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
        else:
            if self.model.config.mm_use_im_start_end:
                qs = image_token_se + "\n" + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
        conv = conv_templates[copy.deepcopy(self.conv_mode)].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        cls_names = get_image_category_from_llava([img], self.llava_tokenizer, self.model, self.image_processor,
                                                  conv.get_prompt(), image_size=None)
        return cls_names[0]

    @torch.no_grad()
    def infer(self, b_patches):
        """
        infer logits from image patches
        """
        image_processed = self.img_processor(b_patches, return_tensors="pt").to(self.image_model.device)
        image_out = self.image_model(**image_processed)

        # normalized features
        image_embeds = image_out.image_embeds / image_out.image_embeds.norm(p=2, dim=-1, keepdim=True)

        # cosine similarity as logits
        # logits_per_text = torch.matmul(self.class_embeddings, image_embeds.t())  # * logit_scale
        # logits_per_image = logits_per_text.t()

        return image_embeds

    def _extract_patches(self, img, patch_size=32, stride=2) -> torch.Tensor:
        patches = img.unfold(2, patch_size, int(patch_size // stride)).unfold(
            3, patch_size, int(patch_size // stride)).flatten(2, 3)
        return patches

    def _process_patches(self, patches):
        patches = patches.permute(0, 2, 1, 3, 4).flatten(0, 1)  # B, C, npatch, hp, wp -> B*npatches C h w
        return [self.to_PIL(patch) for patch in patches]

    def _group_patches(self, patches, output_shape: tuple) -> torch.Tensor:
        """
        ClipConv patch grouping
        Note: this assumes patches are from a square image
        """
        assert len(output_shape) == 4, "output_shape should be 4D"
        B, C, H, W = output_shape
        patches = patches.reshape(B, -1, C)
        num_patches = patches.shape[1]
        num_patches_w = num_patches_h = int(num_patches ** 0.5)
        patches = patches.reshape(B, num_patches_h, num_patches_w, C).permute(0, 3, 1, 2)  # B C H W

        mask = F.interpolate(patches, size=(H, W), mode="nearest")

        return mask


if __name__ == '__main__':
    from libs.vis_utils import visualize_feature_map
    from PIL import Image

    root = './test1.jpg'
    image = Image.open(root).convert("RGB")
    from hydra import compose, initialize

    initialize(config_path="../configs", version_base=None)
    cfg = compose(config_name='cfg_scannet_clip.yaml')
    model = CLIPMeta(cfg)
    img_embeds = model(image)
    print(img_embeds.shape)
    visualize_feature_map(img_embeds.flatten(start_dim=0, end_dim=1).cpu().numpy(), 240, 320)
