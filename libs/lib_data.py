import copy
import math
import os

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
