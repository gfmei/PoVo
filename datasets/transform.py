# function to preprocess the image
import os

import numpy as np
import torch
from torchvision.transforms import transforms, InterpolationMode


def transform_image(image, image_size):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])
    image = transform(image)
    return image


def normalize_pc(points):
    centroid = torch.mean(points, dim=0)
    points -= centroid
    furthest_distance = torch.max(torch.sqrt(torch.sum(torch.abs(points) ** 2, dim=-1)), dim=0)[0]
    points /= furthest_distance

    return centroid, furthest_distance, points


def uvd2xyz(depth, K, extrinsic, depth_trunc=np.inf):
    """
    depth: of shape H, W
    K: 3, 3
    extrinsic: 4, 4
    return points: of shape H, W, 3
    """
    depth[depth > depth_trunc] = 0
    H, W = depth.shape
    fx = K[0][0]
    fy = K[1][1]
    cx = K[0][2]
    cy = K[1][2]
    x = np.arange(0, W) - cx
    y = np.arange(0, H) - cy
    xx, yy = np.meshgrid(x, y)
    points = np.stack([xx, yy, np.ones_like(xx)], axis=-1)
    points = points * depth[..., None]
    points[..., 0] /= fx
    points[..., 1] /= fy
    points = points.reshape(-1, 3)
    points = np.concatenate([points, np.ones_like(points[:, :1])], axis=-1)
    points = points @ np.linalg.inv(extrinsic).T
    points = points[:, :3].reshape(H, W, 3)
    return points


def make_intrinsic(fx, fy, mx, my):
    '''Create camera intrinsics.'''

    intrinsic = np.eye(4)
    intrinsic[0][0] = fx
    intrinsic[1][1] = fy
    intrinsic[0][2] = mx
    intrinsic[1][2] = my
    return intrinsic


def adjust_intrinsic(intrinsic, intrinsic_image_dim, image_dim):
    if intrinsic_image_dim == image_dim:
        return intrinsic

    intrinsic_return = np.copy(intrinsic)

    height_after = image_dim[1]
    height_before = intrinsic_image_dim[1]
    height_ratio = height_after / height_before

    width_after = image_dim[0]
    width_before = intrinsic_image_dim[0]
    width_ratio = width_after / width_before

    if width_ratio >= height_ratio:
        resize_height = height_after
        resize_width = height_ratio * width_before

    else:
        resize_width = width_after
        resize_height = width_ratio * height_before

    intrinsic_return[0, 0] *= float(resize_width) / float(width_before)
    intrinsic_return[1, 1] *= float(resize_height) / float(height_before)
    # account for cropping/padding here
    intrinsic_return[0, 2] *= float(resize_width - 1) / float(width_before - 1)
    intrinsic_return[1, 2] *= float(resize_height - 1) / float(height_before - 1)

    return intrinsic_return


class PointCloudToImageMapper(object):
    def __init__(self, image_dim, visibility_threshold=0.25, cut_bound=0, intrinsics=None, eps=1e-8):

        self.image_dim = image_dim
        self.vis_thres = visibility_threshold
        self.cut_bound = cut_bound
        self.intrinsics = intrinsics
        self.eps = eps

    def compute_mapping(self, camera_to_world, coords, depth=None, intrinsic=None):
        """
        :param camera_to_world: 4 x 4
        :param coords: N x 3 format
        :param depth: H x W format
        :param intrinsic: 3x3 format
        :return: mapping, N x 3 format, (H,W,mask)
        """
        if self.intrinsics is not None:  # global intrinsics
            intrinsic = self.intrinsics

        mapping = np.zeros((3, coords.shape[0]), dtype=int)
        coords_new = np.concatenate([coords, np.ones([coords.shape[0], 1])], axis=1).T
        assert coords_new.shape[0] == 4, "[!] Shape error"

        world_to_camera = np.linalg.inv(camera_to_world)
        p = np.matmul(world_to_camera, coords_new)
        p[2][np.abs(p[2]) < self.eps] = self.eps
        p[0] = (p[0] * intrinsic[0][0]) / p[2] + intrinsic[0][2]
        p[1] = (p[1] * intrinsic[1][1]) / p[2] + intrinsic[1][2]
        pi = np.round(p).astype(int)  # simply round the projected coordinates
        inside_mask = (pi[0] >= self.cut_bound) * (pi[1] >= self.cut_bound) * (
                pi[0] < self.image_dim[0] - self.cut_bound) * (pi[1] < self.image_dim[1] - self.cut_bound)
        if depth is not None:
            depth_cur = depth[pi[1][inside_mask], pi[0][inside_mask]]
            occlusion_mask = np.abs(depth[pi[1][inside_mask], pi[0][inside_mask]]
                                    - p[2][inside_mask]) <= self.vis_thres * depth_cur

            inside_mask[inside_mask == True] = occlusion_mask
        else:
            front_mask = p[2] > 0  # make sure the depth is in front
            inside_mask = front_mask * inside_mask
        mapping[0][inside_mask] = pi[1][inside_mask]
        mapping[1][inside_mask] = pi[0][inside_mask]
        mapping[2][inside_mask] = 1

        return mapping.T


def get_matterport_camera_data(data_path, locs_in, args):
    '''Get all camera view related information of Matterport3D.'''

    # find bounding box of the current region
    bbox_l = locs_in.min(axis=0)
    bbox_h = locs_in.max(axis=0)

    building_name = data_path.split('/')[-1].split('_')[0]
    scene_id = data_path.split('/')[-1].split('.')[0]

    scene = os.path.join(args.data_root_2d, building_name)
    img_names, intrinsics, extrinsics = obtain_intr_extr_matterport(scene)

    cam_loc = extrinsics[:, :3, -1]
    ind_in_scene = (cam_loc[:, 0] > bbox_l[0]) & (cam_loc[:, 0] < bbox_h[0]) & \
                   (cam_loc[:, 1] > bbox_l[1]) & (cam_loc[:, 1] < bbox_h[1]) & \
                   (cam_loc[:, 2] > bbox_l[2]) & (cam_loc[:, 2] < bbox_h[2])

    img_names_in = img_names[ind_in_scene]
    intrinsics_in = intrinsics[ind_in_scene]
    extrinsics_in = extrinsics[ind_in_scene]
    num_img = len(img_names_in)

    # some regions have no views inside, we consider it differently for test and train/val
    if args.split == 'test' and num_img == 0:
        print('no views inside {}, take the nearest 100 images to fuse'.format(scene_id))
        # ! take the nearest 100 views for feature fusion of regions without inside views
        centroid = (bbox_l + bbox_h) / 2
        dist_centroid = np.linalg.norm(cam_loc - centroid, axis=-1)
        ind_in_scene = np.argsort(dist_centroid)[:100]
        img_names_in = img_names[ind_in_scene]
        intrinsics_in = intrinsics[ind_in_scene]
        extrinsics_in = extrinsics[ind_in_scene]
        num_img = 100

    img_names_in = img_names_in.tolist()

    return intrinsics_in, extrinsics_in, img_names_in, scene_id, num_img
