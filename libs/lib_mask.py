import copy

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import ImageDraw
from PIL import Image
from PIL.Image import Resampling
from skimage import img_as_ubyte
from skimage.segmentation import slic
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from torch import nn


class SamMaskGenerator(nn.Module):
    def __init__(self, image_size, model_type="vit_b", checkpoint_path=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        sam.to(device=self.device)
        self.mask_generator = SamAutomaticMaskGenerator(sam)
        self.image_size = image_size

    @staticmethod
    def generate_boxed_images(pil_img, sam_rest, flag=False):
        bounding_boxes = []
        image_list = list()
        for rest in sam_rest:
            x, y, w, h = rest['bbox']
            mask = rest['segmentation']
            bounding_boxes.append(mask)
            img_copy = copy.deepcopy(pil_img)
            draw = ImageDraw.Draw(img_copy)
            draw.rectangle([x, y, x + w, y + h], outline="red", width=3)
            image_list.append(img_copy)
            if flag:
                img_copy.show()
        return image_list, bounding_boxes

    def forward(self, pil_img, flag=False):
        color_image = np.array(pil_img)
        color_image = cv2.resize(color_image, self.image_size)
        result = self.mask_generator.generate(color_image)
        return self.generate_boxed_images(pil_img, result, flag)


def save_boxe_img(img, bbox, name):
    img_copy = img.copy()
    draw = ImageDraw.Draw(img_copy)
    xmin, ymin, xmax, ymax = bbox
    draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=3)
    img_copy.save('{}/{}.png'.format('./', 'box' + str(name)), format='PNG')


def gen_masked_imgs(image_pil, n_segments=12, flag=False):
    # Apply SLIC and obtain the segment labels
    pil_size = image_pil.size
    image = np.array(image_pil)
    segments_slic = slic(image, n_segments=n_segments, compactness=10, sigma=1.0, start_label=1)
    # Visualize the segments
    # segmented_image = label2rgb(segments_slic, image, kind='avg')
    # Creating binary masks for each superpixel
    unique_segments = np.unique(segments_slic)
    # Calculate bounding boxes
    bounding_boxes = []
    image_list = list()
    for i, seg_val in enumerate(unique_segments):
        mask = segments_slic == seg_val
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]
        bounding_boxes.append(mask)
        img = copy.deepcopy(image_pil)
        if flag:
            save_boxe_img(image_pil, (xmin, ymin, xmax, ymax), i)
        cropped_image = img.crop((xmin, ymin, xmax, ymax))
        # Resize the cropped image to the target size
        resized_image = cropped_image.resize(pil_size, Resampling.LANCZOS)
        # resized_image.show()
        image_list.append(resized_image)

    return image_list, bounding_boxes


def get_boxed_images(image_pil, n_segments=16, flag=False):
    # image = io.imread(root)
    # Apply SLIC and obtain the segment labels
    image = np.array(image_pil)
    segments_slic = slic(image, n_segments=n_segments, compactness=10, sigma=1.0, start_label=1)
    # Visualize the segments
    # segmented_image = label2rgb(segments_slic, image, kind='avg')
    # Creating binary masks for each superpixel
    unique_segments = np.unique(segments_slic)
    bounding_boxes = []
    image_list = list()
    for i , seg_val in enumerate(unique_segments):
        mask = segments_slic == seg_val
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]
        bounding_boxes.append(mask)
        img_copy = copy.deepcopy(image_pil)
        draw = ImageDraw.Draw(img_copy)
        draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=3)
        image_list.append(img_copy)
        # if flag:
        #     img_copy.save(f'scan_result/pcd0_image{i}.png')

    return image_list, bounding_boxes


def assign_region_feature_to_image(feat_list, box_list, image_size):
    device = feat_list[0].device
    features = torch.zeros(image_size[1], image_size[0], feat_list[0].size(-1)).to(device)
    w_sum = torch.zeros(image_size[1], image_size[0]).to(device)
    for i, box in enumerate(box_list):
        features[box == True] = feat_list[i]
        w_sum += torch.from_numpy(box).to(device).clip(min=1)
    w_sum = w_sum.clip(min=1)
    return features / w_sum.unsqueeze(-1)


def img_feats_interpolate(feats, size, interpolate_type='bilinear'):
    # Reshape the tensor to [N, C, H, W]. In this case, N=1 since it's a single image
    if feats.dim() == 3:
        feats = feats.unsqueeze(0)
    # print(feats.shape)
    # Specify the scale_factor to resize by (2x in both H and W dimensions)
    output_tensor = F.interpolate(feats, size, mode=interpolate_type)
    # output_tensor = output_tensor.squeeze(0).permute(1, 2, 0)
    output_tensor = output_tensor.squeeze(0)
    return output_tensor


def draw_k2_boxes(image, k, gap=2):
    # Load the image
    # image = cv2.imread(image_path)
    # Get image dimensions
    height, width = image.shape[:2]
    # Adjust dimensions to account for gaps
    total_gap_width = (k + 1) * gap  # Total width consumed by gaps
    total_gap_height = (k + 1) * gap  # Total height consumed by gaps
    box_width = (width - total_gap_width) // k
    box_height = (height - total_gap_height) // k

    # Generate random colors for the boxes
    if k == 2:
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]  # Blue, Green, Red, Cyan
    else:
        colors = [(np.random.randint(255), np.random.randint(255), np.random.randint(255)) for _ in range(k ** 2)]

    # Draw the boxes with gaps
    for row in range(k):
        for col in range(k):
            top_left_corner = (col * (box_width + gap) + gap, row * (box_height + gap) + gap)
            bottom_right_corner = (top_left_corner[0] + box_width, top_left_corner[1] + box_height)
            cv2.rectangle(image, top_left_corner, bottom_right_corner, colors[row * k + col], thickness=2)
    # Display the result
    # cv2.imshow(f'Image with {k**2} Boxes', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    #
    # # Save the result to a file
    # # cv2.imwrite(output_path, image)
    return image


def draw_k2_boxes_with_gaps_pil(image, k=2, gap=2):
    # Load the image
    draw = ImageDraw.Draw(image)

    # Get image dimensions
    width, height = image.size

    # Adjust dimensions to account for gaps
    total_gap_width = (k + 1) * gap  # Total width consumed by gaps
    total_gap_height = (k + 1) * gap  # Total height consumed by gaps
    box_width = (width - total_gap_width) // k
    box_height = (height - total_gap_height) // k

    # Generate random colors for the boxes
    if k == 2:
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]  # Blue, Green, Red, Cyan
    else:
        colors = [(np.random.randint(255), np.random.randint(255), np.random.randint(255)) for _ in range(k ** 2)]
    # Draw the boxes with gaps
    for row in range(k):
        for col in range(k):
            top_left_corner = (col * (box_width + gap) + gap, row * (box_height + gap) + gap)
            bottom_right_corner = (top_left_corner[0] + box_width, top_left_corner[1] + box_height)
            draw.rectangle([top_left_corner, bottom_right_corner], outline=colors[row * k + col], width=2)
    # Display the image
    image.show()
    return image


def get_masks_from_pil(image, n_seg=250):
    image_np = np.array(image)
    # Generate superpixels
    segments = slic(image_np, n_segments=n_seg, compactness=10, sigma=1)
    # Collect superpixel images
    superpixel_images = []
    for segment_id in np.unique(segments):
        mask = segments == segment_id
        masked_image = img_as_ubyte(mask[:, :, None] * image_np)
        superpixel_images.append(Image.fromarray(masked_image))

    return segments, superpixel_images




