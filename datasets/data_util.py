import copy
import math

import numpy as np
import torch


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
