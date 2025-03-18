import copy
import math
import os
from typing import List, Optional, Dict, Tuple

import cv2
import numpy as np
import torch


def load_instance_pcd(root_path, scene_id, split='val'):
    data_root_3d = os.path.join(root_path, 'instance200_gt', split, scene_id + '_inst_nostuff.pth')
    pcds = torch.load(data_root_3d)
    points, colors, labels, instance_ids = pcds[0], pcds[1], pcds[2], pcds[3]
    return points, colors, labels, instance_ids


def make_intrinsic(fx, fy, mx, my):
    '''Create camera intrinsics.'''
    intrinsic = np.eye(4)
    intrinsic[0][0] = fx
    intrinsic[1][1] = fy
    intrinsic[0][2] = mx
    intrinsic[1][2] = my
    return intrinsic


def adjust_intrinsic(intrinsic, intrinsic_image_dim, image_dim):
    '''Adjust camera intrinsics.'''

    if intrinsic_image_dim == image_dim:
        return intrinsic
    resize_width = int(math.floor(image_dim[1] * float(
        intrinsic_image_dim[0]) / float(intrinsic_image_dim[1])))
    intrinsic[0, 0] *= float(resize_width) / float(intrinsic_image_dim[0])
    intrinsic[1, 1] *= float(image_dim[1]) / float(intrinsic_image_dim[1])
    # account for cropping here
    intrinsic[0, 2] *= float(image_dim[0] - 1) / float(intrinsic_image_dim[0] - 1)
    intrinsic[1, 2] *= float(image_dim[1] - 1) / float(intrinsic_image_dim[1] - 1)
    return intrinsic


def get_intrinsic(img_dim, intrinsic_image_dim=None, intrinsic=None):
    if intrinsic_image_dim is None:
        intrinsic_image_dim = [640, 480]
    if intrinsic is None:
        fx = 577.870605
        fy = 577.870605
        mx = 319.5
        my = 239.5
        # calculate image pixel-3D points correspondances
        intrinsic = make_intrinsic(fx=fx, fy=fy, mx=mx, my=my)
    intrinsic = adjust_intrinsic(intrinsic, intrinsic_image_dim, image_dim=img_dim)

    return intrinsic


def label_revise(gt_cls, pred_cls, score, threshold=0.6):
    X = copy.deepcopy(gt_cls)
    Y = pred_cls
    val, index = torch.max(score, dim=1)  # Max value and index in Y for each element in X
    # Threshold for replacement
    # Assuming val and index are tensors, convert val to a list for direct comparison
    val_list = val.tolist()
    # Replace elements in X based on the threshold
    X_updated = [Y[index[i].item()] if val_list[i] > threshold else X[i] for i in range(len(X))]
    return X_updated



def num_to_natural(group_ids, void_number=-1):
    """
    code credit: SAM3D
    """
    if (void_number == -1):
        # [-1,-1,0,3,4,0,6] -> [-1,-1,0,1,2,0,3]
        if np.all(group_ids == -1):
            return group_ids
        array = group_ids.copy()

        unique_values = np.unique(array[array != -1])
        mapping = np.full(np.max(unique_values) + 2, -1)
        # map ith(start from 0) group_id to i
        mapping[unique_values + 1] = np.arange(len(unique_values))
        array = mapping[array + 1]

    elif (void_number == 0):
        # [0,3,4,0,6] -> [0,1,2,0,3]
        if np.all(group_ids == 0):
            return group_ids
        array = group_ids.copy()

        unique_values = np.unique(array[array != 0])
        mapping = np.full(np.max(unique_values) + 2, 0)
        mapping[unique_values] = np.arange(len(unique_values)) + 1
        array = mapping[array]
    else:
        raise Exception("void_number must be -1 or 0")

    return array


def num_to_natural_torch(group_ids, void_number=-1):
    """
    Convert group ids to natural numbers, handling void numbers as specified.
    code credit: SAM3D
    """
    group_ids_tensor = group_ids.long()
    device = group_ids_tensor.device

    if void_number == -1:
        # [-1,-1,0,3,4,0,6] -> [-1,-1,0,1,2,0,3]
        if torch.all(group_ids_tensor == -1):
            return group_ids_tensor
        array = group_ids_tensor.clone()

        unique_values = torch.unique(array[array != -1])
        mapping = torch.full((torch.max(unique_values) + 2,), -1, dtype=torch.long, device=device)
        # map ith (start from 0) group_id to i
        mapping[unique_values + 1] = torch.arange(len(unique_values), dtype=torch.long, device=device)
        array = mapping[array + 1]

    elif void_number == 0:
        # [0,3,4,0,6] -> [0,1,2,0,3]
        if torch.all(group_ids_tensor == 0):
            return group_ids_tensor
        array = group_ids_tensor.clone()

        unique_values = torch.unique(array[array != 0])
        mapping = torch.full((torch.max(unique_values) + 2,), 0, dtype=torch.long, device=device)
        mapping[unique_values] = torch.arange(len(unique_values), dtype=torch.long, device=device) + 1
        array = mapping[array]
    else:
        raise Exception("void_number must be -1 or 0")

    return array


def resize_mask(mask, img_size):
    # Assume the mask is a 2D boolean numpy array
    resized_segmentation = cv2.resize(
        mask.astype(np.float32), img_size, interpolation=cv2.INTER_NEAREST) > 0.5
    return np.uint8(resized_segmentation * 255)



class BoundingBox:
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @property
    def xyxy(self) -> List[float]:
        return [self.xmin, self.ymin, self.xmax, self.ymax]


class DetectionResult:
    score: float
    label: str
    box: BoundingBox
    mask: Optional[np.array] = None

    @classmethod
    def from_dict(cls, detection_dict: Dict) -> 'DetectionResult':
        return cls(score=detection_dict['score'],
                   label=detection_dict['label'],
                   box=BoundingBox(xmin=detection_dict['box']['xmin'],
                                   ymin=detection_dict['box']['ymin'],
                                   xmax=detection_dict['box']['xmax'],
                                   ymax=detection_dict['box']['ymax']))


def get_boxes(results: DetectionResult) -> List[List[List[float]]]:
    boxes = []
    for result in results:
        xyxy = result.box.xyxy
        boxes.append(xyxy)

    return [boxes]


def mask_to_polygon(mask: np.ndarray) -> List[List[int]]:
    # Find contours in the binary mask
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Find the contour with the largest area
    largest_contour = max(contours, key=cv2.contourArea)
    # Extract the vertices of the contour
    polygon = largest_contour.reshape(-1, 2).tolist()

    return polygon


def polygon_to_mask(polygon: List[Tuple[int, int]], image_shape: Tuple[int, int]) -> np.ndarray:
    """
    Convert a polygon to a segmentation mask.
    Args:
    - polygon (list): List of (x, y) coordinates representing the vertices of the polygon.
    - image_shape (tuple): Shape of the image (height, width) for the mask.
    Returns:
    - np.ndarray: Segmentation mask with the polygon filled.
    """
    # Create an empty mask
    mask = np.zeros(image_shape, dtype=np.uint8)
    # Convert polygon to an array of points
    pts = np.array(polygon, dtype=np.int32)
    # Fill the polygon with white color (255)
    cv2.fillPoly(mask, [pts], color=(255,))

    return mask


def refine_masks(masks: torch.BoolTensor, polygon_refinement: bool = False) -> List[np.ndarray]:
    masks = masks.cpu().float()
    masks = masks.permute(0, 2, 3, 1)
    masks = masks.mean(axis=-1)
    masks = (masks > 0).int()
    masks = masks.numpy().astype(np.uint8)
    masks = list(masks)

    if polygon_refinement:
        for idx, mask in enumerate(masks):
            shape = mask.shape
            polygon = mask_to_polygon(mask)
            mask = polygon_to_mask(polygon, shape)
            masks[idx] = mask

    return masks

