import argparse
import os
from glob import glob
from os.path import join

import cv2
import numpy as np
import torch
from PIL import Image
from hydra import initialize, compose
from torch.utils.data import Dataset
from tqdm import tqdm

from datasets.meta_data.label_constants import MATTERPORT_LABELS_160
from datasets.voxelizer import Voxelizer
from libs.lib_mask import img_feats_interpolate
from libs.o3d_util import normal_estimation
from models.modules import CLIPMeta, PatchCLIP, CLIPText
from transform import PointCloudToImageMapper


def obtain_intr_extr_matterport(scene):
    """Obtain the intrinsic and extrinsic parameters of Matterport3D."""
    img_dir = os.path.join(scene, 'color')
    pose_dir = os.path.join(scene, 'pose')
    intr_dir = os.path.join(scene, 'intrinsic')
    img_names = sorted(glob(img_dir + '/*.jpg'))

    intrinsics = []
    extrinsics = []
    for img_name in img_names:
        name = img_name.split('/')[-1][:-4]

        extrinsics.append(np.loadtxt(os.path.join(pose_dir, name + '.txt')))
        intrinsics.append(np.loadtxt(os.path.join(intr_dir, name + '.txt')))

    intrinsics = np.stack(intrinsics, axis=0)
    extrinsics = np.stack(extrinsics, axis=0)
    img_names = np.asarray(img_names)

    return img_names, intrinsics, extrinsics


def get_matterport_camera_data(data_path, data_root_2d, locs_in, split):
    """Get all camera view related information of Matterport3D."""
    # find bounding box of the current region
    bbox_l = locs_in.min(axis=0)
    bbox_h = locs_in.max(axis=0)
    building_name = data_path.split('/')[-1].split('_')[0]
    scene_id = data_path.split('/')[-1].split('.')[0]
    scene = os.path.join(data_root_2d, building_name)
    img_names, intrinsics, extrinsics = obtain_intr_extr_matterport(scene)

    cam_loc = extrinsics[:, :3, -1]
    ind_in_scene = (cam_loc[:, 0] > bbox_l[0]) & (cam_loc[:, 0] < bbox_h[0]) & (cam_loc[:, 1] > bbox_l[1]) & (
            cam_loc[:, 1] < bbox_h[1]) & (cam_loc[:, 2] > bbox_l[2]) & (cam_loc[:, 2] < bbox_h[2])

    img_names_in = img_names[ind_in_scene]
    intrinsics_in = intrinsics[ind_in_scene]
    extrinsics_in = extrinsics[ind_in_scene]
    num_img = len(img_names_in)

    # some regions have no views inside, we consider it differently for test and train/val
    if split == 'test' and num_img == 0:
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


class Matterport(Dataset):
    # Augmentation arguments
    SCALE_AUGMENTATION_BOUND = (0.9, 1.1)
    ROTATION_AUGMENTATION_BOUND = ((-np.pi / 64, np.pi / 64), (-np.pi / 64, np.pi / 64), (-np.pi, np.pi))
    TRANSLATION_AUGMENTATION_RATIO_BOUND = ((-0.2, 0.2), (-0.2, 0.2), (0, 0))
    ELASTIC_DISTORT_PARAMS = ((0.2, 0.4), (0.8, 1.6))
    ROTATION_AXIS = 'z'
    LOCFEAT_IDX = 2

    def __init__(self, root='data', voxel_size=0.05, img_size=(640, 512),
                 vis_name='llava',
                 cut_num_pixel_boundary=10,  # do not use the features on the image boundary
                 split='train', aug=False, identifier=1233,
                 visibility_threshold=0.02,
                 depth_scale=4000.0,
                 num_classes=160,
                 device='auto'):
        super(Matterport, self).__init__()

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

        self.point2img_mapper = PointCloudToImageMapper(
            image_dim=img_size, intrinsics=None,
            visibility_threshold=visibility_threshold,
            cut_bound=cut_num_pixel_boundary)

        self.voxel_size = voxel_size
        self.voxelizer = Voxelizer(
            voxel_size=voxel_size,
            clip_bound=None,
            use_augmentation=True,
            scale_augmentation_bound=self.SCALE_AUGMENTATION_BOUND,
            rotation_augmentation_bound=self.ROTATION_AUGMENTATION_BOUND,
            translation_augmentation_ratio_bound=self.TRANSLATION_AUGMENTATION_RATIO_BOUND)

        self.vis_name = vis_name
        initialize(config_path="../configs", version_base=None)
        cfg = compose(config_name='cfg_scannet_clip.yaml')

        if vis_name == 'denseclip':
            self.vis_encoder = CLIPMeta(cfg)
        elif vis_name == 'patchclip':
            self.vis_encoder = PatchCLIP(cfg)
        # elif vis_name == 'dinov2':
        #     model_path = "facebookresearch/dinov2"
        #     model_name = "dinov2_vitl14"
        #     self.vis_encoder = DINOV2(model_path, model_name, device=device)
        else:
            self.vis_encoder = CLIPText(cfg)

        data_root_3d = join(root, 'matterport_3d_{}'.format(num_classes))
        self.data_root_2d = join(root, 'matterport_2d')
        self.split = split
        self.depth_scale = depth_scale
        self.data_paths = glob(join(data_root_3d, split, '*.pth'))
        self.feat_dim = self.vis_encoder.clip_model.projection_dim
        self.img_size = img_size

        ids2cat = list(MATTERPORT_LABELS_160)
        ids2cat.append('otherfurniture')
        self.lb_emd = self.vis_encoder.text_embedding(ids2cat)

    def __getitem__(self, item):
        data_path = self.data_paths[item]
        data = torch.load(data_path)
        raw_points = data[0]
        raw_colors = data[1]
        raw_labels = data[-1]
        # obtain all camera views related information (specificially for Matterport)
        intrinsics, poses, img_dirs, scene_id, num_img = get_matterport_camera_data(
            data_path, self.data_root_2d, raw_points, self.split)
        # keep_features_in_memory = args.keep_features_in_memory
        # load 3D data (point cloud, color and the corresponding labels)
        n_points = raw_points.shape[0]
        if num_img == 0:
            print('no views inside {}'.format(scene_id))
            return 1
        n_points_cur = n_points
        counter = torch.zeros((n_points_cur, 1), device=self.device)
        sum_features = torch.zeros((n_points_cur, self.feat_dim), device=self.device)
        ################ Feature Fusion ###################
        vis_id = torch.zeros((n_points_cur, num_img), dtype=int, device=self.device)
        cls_names = set()
        for img_id, img_dir in enumerate(tqdm(img_dirs)):
            # load pose
            pose = poses[img_id]
            # load per-image intrinsic
            intr = intrinsics[img_id]

            # load depth and convert to meter
            depth_dir = img_dir.replace('color', 'depth')
            _, img_type, yaw_id = img_dir.split('/')[-1].split('_')
            depth_dir = depth_dir[:-8] + 'd' + img_type[1] + '_' + yaw_id[0] + '.png'
            depth = np.array(cv2.imread(depth_dir, cv2.IMREAD_ANYDEPTH)) / self.depth_scale

            # Downsample the depth image using cv2.resize
            if self.img_size[0] != depth.shape[1] or self.img_size[1] != depth.shape[0]:
                depth = cv2.resize(depth, self.img_size, interpolation=cv2.INTER_LINEAR)
            # image = cv2.imread(img_dir, cv2.IMREAD_COLOR)
            image = Image.open(img_dir).convert("RGB")
            if self.img_size[0] != image.size[0] or self.img_size[1] != image.size[1]:
                image = image.resize(self.img_size)

            imgi_feature = self.vis_encoder(image, flag=False).clone().to(self.device, dtype=torch.float32)
            new_names = self.vis_encoder.get_image_level_names(image)

            cls_names.update(new_names)
            imgi_feature = imgi_feature.permute(2, 0, 1)
            if self.img_size[0] != imgi_feature.shape[1] or self.img_size[1] != imgi_feature.shape[0]:
                imgi_feature = img_feats_interpolate(imgi_feature, self.img_size[::-1], interpolate_type='nearest')
            # calculate the 3d-2d mapping based on the depth
            mapping = np.ones([n_points, 4], dtype=int)
            mapping[:, 1:4] = self.point2img_mapper.compute_mapping(pose, raw_points, depth, intr)
            if mapping[:, 3].sum() == 0:  # no points corresponds to this image, skip
                continue
            mapping = torch.from_numpy(mapping).to(self.device)
            mask = mapping[:, 3]
            vis_id[:, img_id] = mask
            vis_id[:, img_id] = mask
            feat_2d = imgi_feature

            feat_2d_3d = feat_2d[:, mapping[:, 1], mapping[:, 2]].permute(1, 0)
            counter[mask != 0] += 1
            sum_features[mask != 0] += feat_2d_3d[mask != 0]

        counter[counter == 0] = 1e-5
        feat_bank = sum_features / counter
        point_ids = torch.unique(vis_id.nonzero(as_tuple=False)[:, 0])
        points = raw_points[point_ids]
        normals = normal_estimation(points, knn=33)
        # colors = raw_colors[point_ids]
        # label_set = set(raw_labels.tolist())
        gt_label = raw_labels[point_ids]
        # num_gt = len(set(gt_label.tolist()))
        # num_pred = len(cls_names)
        # gt_set = set(gt_label.tolist())
        # print([ids2cat[d] for d in gt_set], cls_names)
        llava_emd = self.vis_encoder.text_embedding(list(cls_names))
        cls_names = list(cls_names)
        return points, feat_bank[point_ids], normals, cls_names, self.lb_emd[gt_label], llava_emd, gt_label


def get_args():
    '''Command line arguments.'''

    parser = argparse.ArgumentParser(
        description='Multi-view feature fusion of OpenSeg on Matterport3D.')
    parser.add_argument('--data_dir', type=str, help='Where is the base logging directory')
    parser.add_argument('--output_dir', type=str, help='Where is the base logging directory')
    parser.add_argument('--split', type=str, default='test', help='split: "train"| "val" | "test" ')
    parser.add_argument('--openseg_model', type=str, default='', help='Where is the exported OpenSeg model')
    parser.add_argument('--process_id_range', nargs='+', default=None, help='the id range to process')
    parser.add_argument('--img_feat_dir', type=str, default='', help='the id range to process')

    # Hyper parameters
    parser.add_argument('--hparams', default=[], nargs="+")
    args = parser.parse_args()
    return args
