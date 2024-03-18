import os
import pickle
import random
import sys
from glob import glob
from os.path import join

import cv2
import numpy as np
import torch
from PIL import Image
from hydra import initialize, compose
from torch.utils.data import Dataset
from tqdm import tqdm

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append('../libs')
sys.path.append('../datasets')
sys.path.append('../models')

from data_util import custom_collate_fn
from meta_data.label_constants import MATTERPORT_LABELS_160
from voxelizer import Voxelizer
from libs.lib_mask import img_feats_interpolate
from libs.lib_utils import remove_repeat_words
from libs.o3d_util import normal_estimation
from models.modules import CLIPMeta, PatchCLIP, CLIPText
from transform import PointCloudToImageMapper
from models.segmodel import max_vote


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
    scene = os.path.join(data_root_2d, split, building_name)
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
                 is_orig=True,
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
            use_augmentation=False,
            scale_augmentation_bound=self.SCALE_AUGMENTATION_BOUND,
            rotation_augmentation_bound=self.ROTATION_AUGMENTATION_BOUND,
            translation_augmentation_ratio_bound=self.TRANSLATION_AUGMENTATION_RATIO_BOUND)

        self.vis_name = vis_name
        initialize(config_path="../configs", version_base=None)
        cfg = compose(config_name='cfg_matterport_clip.yaml')

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
        self.lb_emd = self.vis_encoder.text_embedding(ids2cat).to(self.device, dtype=torch.float16)
        self.is_orig = is_orig

    def __getitem__(self, item):
        data_path = self.data_paths[item]
        data = torch.load(data_path)
        raw_points = data[0]
        # raw_colors = data[1]
        raw_labels = data[-1]
        # raw_points, raw_colors, raw_labels = self.voxelizer.voxelize(raw_points, raw_colors, raw_labels)
        # obtain all camera views related information (specificially for Matterport)
        intrinsics, poses, img_dirs, scene_id, num_img = get_matterport_camera_data(
            data_path, self.data_root_2d, raw_points, self.split)
        # keep_features_in_memory = args.keep_features_in_memory
        raw_labels[raw_labels == 255] = 160
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
        # k = min(len(img_dirs), 2)
        # img_dirs = img_dirs[0:k]
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
            imgi_feature = self.vis_encoder(image, flag=True).clone().to(self.device, dtype=torch.float16)
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
        point_ids = torch.unique(vis_id.nonzero(as_tuple=False)[:, 0]).tolist()
        raw_points = torch.tensor(raw_points, dtype=torch.float16).to(self.device)
        points = raw_points[point_ids]
        normals = normal_estimation(points, knn=33)
        # colors = raw_colors[point_ids]
        # label_set = set(raw_labels.tolist())
        points = raw_points[point_ids]
        gt_labels = raw_labels[point_ids]
        cls_names = list(remove_repeat_words(cls_names))
        llava_emd = self.vis_encoder.text_embedding(list(cls_names)).to(self.device, dtype=torch.float16)
        cls_names = list(cls_names)
        feats = feat_bank[point_ids].detach()
        normals, gt_labels = torch.tensor(normals), torch.tensor(gt_labels)

        if self.is_orig:
            return points, feats, normals, cls_names, raw_points, llava_emd, gt_labels.to(self.device)
        pcd_gt_feats = self.lb_emd[gt_labels].to(self.device, dtype=torch.float16)
        return points, feats, normals, cls_names, pcd_gt_feats, llava_emd, gt_labels.to(self.device)

    def __len__(self):
        return len(self.data_paths)


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from libs.vis_utils import creat_labeled_point_cloud, get_colored_point_cloud_pca_sep, draw_superpoints

    # data_root = '/data/disk1/data/Matterport'
    data_root = '/storage2/TEV/datasets/Matterport'
    test_data = Matterport(data_root, vis_name='textclip', split='test', img_size=(640, 512))
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=True, collate_fn=custom_collate_fn)

    for i, data in enumerate(test_dataloader):
        pcd, feat, normal, llava_cls_names, opcd, llavalb_emd, gt_labelid = data
        # get_colored_point_cloud_pca_sep(pcd[0].detach().cpu().numpy(), gtlb_emd[0].detach().cpu().numpy(), 'result/gtpca')
        # get_colored_point_cloud_from_labels(pcd[0].detach().cpu().numpy(), gamma[0].detach().cpu().numpy(), name='clus')
        # draw_point_cloud(rpoints[0].cpu().numpy(), None, f'result/org{str(i)}')
        pdlb_embed = llavalb_emd[0]
        # gt_ids = torch.argmax(torch.einsum('md,nd->mn', gtpcd_emd[0],
        #                                    llavalb_emd[0].to(gtpcd_emd[0])), dim=-1).flatten().tolist()
        # creat_labeled_point_cloud(pcd[0].detach().cpu().numpy(), gt_ids, f'result/gt{str(i)}')
        # score = torch.einsum('md,nd->mn', test_data.lb_emd, llavalb_emd[0])
        ids2cat = list(MATTERPORT_LABELS_160)
        ids2cat.append('otherfurniture')
        # uni_ids = set(gt_labelid[0].tolist())
        print(set(llava_cls_names[0]).intersection(set(ids2cat)))
        get_colored_point_cloud_pca_sep(pcd[0].detach().cpu().numpy(), feat[0].detach().cpu().numpy(),
                                        f'result/pca{str(i)}')
        draw_superpoints(pcd[0].detach().cpu().numpy(), normal[0].detach().cpu().numpy(), f'result/spts{str(i)}')
        pred_scores = 1 + torch.einsum('md,nd->mn', feat[0], pdlb_embed.to(feat[0]))
        # d_pcd, pred_labels1 = max_vote(pcd[0].to(pred_scores), normal[0].to(pred_scores), pred_scores.to(pred_scores))
        pred_labels = torch.argmax(pred_scores, dim=-1)
        creat_labeled_point_cloud(pcd[0].detach().cpu().numpy(), pred_labels.flatten().tolist(), f'result/pred{str(i)}')
        # Save using pickle
        print('The {}th point cloud has been processed'.format(i))
        with open(f'result/pcd_{i}.pickle', 'wb') as f:
            pickle.dump({'pcd': pcd[0], 'pred_names': [llava_cls_names[0][ids] for ids in pred_labels],
                         # 'spts': [llava_cls_names[0][ids] for ids in pred_labels],
                         'gt_names': [ids2cat[ids] for ids in gt_labelid[0].tolist()]}, f)
        # # print(update_names, ids2cat)
        # print(i, data[0].shape)
        #     # gamma = fusion_wkeans([pcd.float(), normal.float()],
        #     #                       [None, 1], n_clus=20, iters=10, is_prob=False, idx=0)[0]
        #     print(pcd.shape, feat.shape, txt_emd.shape)
        #     # label_np = list(set(labels[0].to(torch.int32).tolist()))
        #     # label_np = set(labels[0].to(torch.int32).tolist())
        #     # color = colors[0].detach().cpu().numpy()
        # draw_point_cloud(org_pcd[0].detach().cpu().numpy(), None, 'orig')
        # draw_point_cloud(superpoints[0].detach().cpu().numpy(), None, 'spts')
        if i > 5:
            break
