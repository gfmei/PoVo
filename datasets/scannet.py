import copy
import os
import pickle
import random
import sys
from os.path import join

import cv2
import numpy as np
import torch
from PIL import Image
from glob2 import glob
from hydra import compose, initialize
from torch.utils.data import Dataset

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append('../libs')
sys.path.append('../datasets')
sys.path.append('../models')
from voxelizer import Voxelizer
from data_util import make_intrinsic, adjust_intrinsic
from clip_models.DINOv2 import DINOV2
from libs.mask_lib import img_feats_interpolate
from models.modules import CLIPMeta, CLIPText, PatchCLIP
from meta_data.scannet200_constants import CLASS_LABELS_200
from transform import PointCloudToImageMapper


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


class ScanNet(Dataset):
    '''Dataloader for 3D points and labels.'''

    # Augmentation arguments
    SCALE_AUGMENTATION_BOUND = (0.9, 1.1)
    ROTATION_AUGMENTATION_BOUND = ((-np.pi / 64, np.pi / 64), (-np.pi / 64, np.pi / 64), (-np.pi, np.pi))
    TRANSLATION_AUGMENTATION_RATIO_BOUND = ((-0.2, 0.2), (-0.2, 0.2), (0, 0))
    ELASTIC_DISTORT_PARAMS = ((0.2, 0.4), (0.8, 1.6))
    ROTATION_AXIS = 'z'
    LOCFEAT_IDX = 2

    def __init__(self, root='data', voxel_size=0.05, img_size=(48, 48),
                 vis_name='llava',
                 cut_num_pixel_boundary=10,  # do not use the features on the image boundary
                 split='train', aug=False, identifier=1233,
                 visibility_threshold=0.05,
                 depth_scale=1000.0,
                 device='auto'
                 ):
        super().__init__()
        self.split = split
        if split is None:
            split = ''
        self.identifier = identifier
        self.data_dir = root
        if split == 'train':  # for training set, export a chunk of point cloud
            self.n_split_points = 20000
            self.num_rand_file_per_scene = 5
            self.scannet_file_list = self.read_files('../dataset/preprocess/meta_data/scannetv2_train.txt')
        else:  # for the validation set, export the entire point cloud instead of chunks
            self.n_split_points = 2000000
            self.num_rand_file_per_scene = 1
            self.scannet_file_list = self.read_files('../dataset/preprocess/meta_data/scannetv2_val.txt')

        # intrinsics = np.loadtxt(join(root, 'scannet_2d', 'intrinsics.txt'))
        # intrinsics = get_intrinsic(img_size)
        intrinsics = get_intrinsic(img_size, intrinsic=None)
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

        self.vis_name = vis_name

        initialize(config_path="../configs", version_base=None)
        cfg = compose(config_name='cfg_scannet_clip.yaml')

        if vis_name == 'denseclip':
            self.vis_encoder = CLIPMeta(cfg)
        elif vis_name == 'patchclip':
            self.vis_encoder = PatchCLIP(cfg)
        elif vis_name == 'dinov2':
            model_path = "facebookresearch/dinov2"
            model_name = "dinov2_vitl14"
            self.vis_encoder = DINOV2(model_path, model_name, device=device)
        else:
            self.vis_encoder = CLIPText(cfg)
        # if vis_name == 'patchclip':
        #     self.text_encoder = self.vis_encoder
        # else:
        #     self.text_encoder = CLIPText(cfg)
        # self.vis_encoder.to(device)
        self.img_size = img_size
        self.point2img_mapper = PointCloudToImageMapper(
            image_dim=(img_size[0], img_size[1]), intrinsics=intrinsics,
            visibility_threshold=visibility_threshold,
            cut_bound=cut_num_pixel_boundary)
        self.voxel_size = voxel_size
        self.aug = aug
        self.depth_scale = depth_scale

        self.voxelizer = Voxelizer(
            voxel_size=voxel_size,
            clip_bound=None,
            use_augmentation=True,
            scale_augmentation_bound=self.SCALE_AUGMENTATION_BOUND,
            rotation_augmentation_bound=self.ROTATION_AUGMENTATION_BOUND,
            translation_augmentation_ratio_bound=self.TRANSLATION_AUGMENTATION_RATIO_BOUND)
        ids2cat = list(CLASS_LABELS_200)
        ids2cat.append('otherfurniture')
        self.lb_emd = self.vis_encoder.text_embedding(ids2cat)

    @staticmethod
    def read_files(file):
        f = open(file)
        lines = f.readlines()
        name_list = [line.split('.')[0].strip() for line in lines]
        f.close()
        return name_list

    def __getitem__(self, index_long):
        index = index_long % len(self.scannet_file_list)
        scene_id = self.scannet_file_list[index]
        scene_name = scene_id + '.pth'
        data_root_3d = join(self.data_dir, 'scannet_3d', self.split, scene_name)
        data_root_2d = join(self.data_dir, 'scannet_2d')
        img_dirs = sorted(glob(join(data_root_2d, self.split, scene_id, 'color/*')),
                          key=lambda x: int(os.path.basename(x)[:-4]))
        k = min(len(img_dirs), 64)
        img_dirs = random.sample(img_dirs, k)
        num_img = len(img_dirs)
        device = torch.device('cpu')
        # load 3D data (point cloud)
        load_pcd = torch.load(data_root_3d)
        raw_pcd = load_pcd['coord']
        # raw_colors = load_pcd[1]
        raw_labels = load_pcd['semantic_gt200'].astype(int)
        raw_normals = load_pcd['normal']
        raw_labels[raw_labels == -1] = 200
        # mesh_root_3d = join(self.data_dir, 'scans', scene_id, scene_id + '_vh_clean_2.ply')
        # superpoints = get_superpoint_cloud(mesh_root_3d, np.ascontiguousarray(raw_pcd))
        superpoints = load_pcd["instance_gt"]
        n_points = raw_pcd.shape[0]
        n_points_cur = n_points
        counter = torch.zeros((n_points_cur, 1), device=device)
        feat_dim = self.vis_encoder.clip_model.projection_dim
        sum_features = torch.zeros((n_points_cur, feat_dim), device=device)
        cls_names = set()
        ################ Feature Fusion ###################
        vis_id = torch.zeros((n_points_cur, num_img), dtype=int, device=device)
        for img_id, img_dir in enumerate(img_dirs):
            # load pose
            posepath = img_dir.replace('color', 'pose').replace('.jpg', '.txt')
            pose = np.loadtxt(posepath).astype(float)  # camera2world
            # load depth and convert to meter
            depth = np.array(cv2.imread(img_dir.replace('color', 'depth').replace(
                'jpg', 'png'), cv2.IMREAD_ANYDEPTH)) / self.depth_scale
            # Downsample the depth image using cv2.resize
            if self.img_size[0] != depth.shape[1] or self.img_size[1] != depth.shape[0]:
                depth = cv2.resize(depth, self.img_size, interpolation=cv2.INTER_LINEAR)
            # image = cv2.imread(img_dir, cv2.IMREAD_COLOR)
            image = Image.open(img_dir).convert("RGB")
            if self.img_size[0] != image.size[0] or self.img_size[1] != image.size[1]:
                image = image.resize(self.img_size)
            # calculate the 3d-2d mapping based on the depth
            # if img_id < 6:
            #     flag = True
            #     image.save(f'rgb{img_id}.jpg')
            # else:
            #     flag = False
            imgi_feature = self.vis_encoder(image, flag=False).clone().to(device, dtype=torch.float32)
            new_names = self.vis_encoder.get_image_level_names(image)
            # print(new_names)
            cls_names.update(new_names)
            imgi_feature = imgi_feature.permute(2, 0, 1)
            if self.img_size[0] != imgi_feature.shape[1] or self.img_size[1] != imgi_feature.shape[0]:
                imgi_feature = img_feats_interpolate(imgi_feature, self.img_size[::-1], interpolate_type='nearest')
            mapping = np.ones([n_points, 4], dtype=int)
            mapping[:, 1:4] = self.point2img_mapper.compute_mapping(pose, raw_pcd, depth)
            if mapping[:, 3].sum() == 0:  # no points corresponds to this image, skip
                continue
            mapping = torch.from_numpy(mapping).to(imgi_feature.device)
            mask = mapping[:, 3]
            vis_id[:, img_id] = mask
            feat_2d = imgi_feature

            feat_2d_3d = feat_2d[:, mapping[:, 1], mapping[:, 2]].permute(1, 0)
            counter[mask != 0] += 1
            sum_features[mask != 0] += feat_2d_3d[mask != 0]

        counter[counter == 0] = 1e-5
        feat_bank = sum_features / counter
        point_ids = torch.unique(vis_id.nonzero(as_tuple=False)[:, 0])
        points = raw_pcd[point_ids]
        # normals = normal_estimation(points, knn=33)
        # colors = raw_colors[point_ids]
        # label_set = set(raw_labels.tolist())
        gt_label = raw_labels[point_ids]
        # num_gt = len(set(gt_label.tolist()))
        # num_pred = len(cls_names)
        # gt_set = set(gt_label.tolist())
        # print([ids2cat[d] for d in gt_set], cls_names)
        # lb_emd = self.vis_encoder.text_embedding([ids2cat[lb] for lb in label_set])
        llava_emd = self.vis_encoder.text_embedding(list(cls_names))
        cls_names = list(cls_names)
        return points, feat_bank[point_ids], raw_normals[point_ids], cls_names, self.lb_emd[
            gt_label], llava_emd, gt_label

    def __len__(self):
        return len(self.scannet_file_list)


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from libs.vis_utils import creat_labeled_point_cloud, get_colored_point_cloud_pca_sep

    data_root = '/data/disk1/data/scannet'
    # data_root = '/storage/TEV/datasets/ScanNet'
    test_data = ScanNet(data_root, vis_name='textclip', split='val', img_size=(320, 240))
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=True)

    for i, data in enumerate(test_dataloader):
        pcd, feat, normal, llava_cls_name, gtpcd_emd, llavalb_emd, gt_labelid = data
        # get_colored_point_cloud_pca_sep(pcd[0].detach().cpu().numpy(), gtlb_emd[0].detach().cpu().numpy(), 'result/gtpca')
        # get_colored_point_cloud_from_labels(pcd[0].detach().cpu().numpy(), gamma[0].detach().cpu().numpy(), name='clus')
        creat_labeled_point_cloud(pcd[0].detach().cpu().numpy(), gt_labelid[0].detach().cpu().numpy(),
                                  f'result/gt{str(i)}')
        score = torch.einsum('md,nd->mn', test_data.lb_emd, llavalb_emd[0])
        ids2cat = list(CLASS_LABELS_200)
        ids2cat.append('otherfurniture')
        # print(llava_cls_name)
        llava_cls_names = [item[0] for item in llava_cls_name]
        # print(llava_cls_names, len(ids2cat), len(llavalb_emd[0]))
        # update_names = label_revise(ids2cat, llava_cls_names, score)
        # print(len(update_names[0]), len(ids2cat), len(llavalb_emd[0]))
        get_colored_point_cloud_pca_sep(pcd[0].detach().cpu().numpy(), feat[0].detach().cpu().numpy(),
                                        f'result/pca{str(i)}')
        pred_labels = torch.argmax(torch.einsum('md,nd->mn', feat[0], llavalb_emd[0]), dim=-1).flatten().tolist()
        creat_labeled_point_cloud(pcd[0].detach().cpu().numpy(), pred_labels, f'result/pred{str(i)}')
        # Save using pickle
        print('this is number', i)
        with open(f'result/pcd_{i}.pickle', 'wb') as f:
            pickle.dump({'tensor': pcd[0], 'pred_names': [llava_cls_names[ids] for ids in pred_labels],
                         'gt_names': [ids2cat[ids] for ids in gt_labelid[0].tolist()]}, f)
        # print(update_names, ids2cat)
        # print(i, data[0].shape)
        #     # gamma = fusion_wkeans([pcd.float(), normal.float()],
        #     #                       [None, 1], n_clus=20, iters=10, is_prob=False, idx=0)[0]
        #     print(pcd.shape, feat.shape, txt_emd.shape)
        #     # label_np = list(set(labels[0].to(torch.int32).tolist()))
        #     # label_np = set(labels[0].to(torch.int32).tolist())
        #     # color = colors[0].detach().cpu().numpy()
        # draw_point_cloud(org_pcd[0].detach().cpu().numpy(), None, 'orig')
        # draw_point_cloud(superpoints[0].detach().cpu().numpy(), None, 'spts')
        if i > 10:
            break
