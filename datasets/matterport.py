import argparse
import os
from glob import glob
from os.path import join

import imageio
import numpy as np
import tensorflow as tf2
import torch
from torch.utils.data import Dataset
from tqdm import tqdm, trange

from transform import PointCloudToImageMapper


def obtain_intr_extr_matterport(scene):
    '''Obtain the intrinsic and extrinsic parameters of Matterport3D.'''

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

        data_root_3d = join(root, 'matterport_3d_{}'.format(num_classes))
        self.data_root_2d = join(root, 'matterport_2d')
        self.split = split
        self.depth_scale = depth_scale
        self.data_paths = glob(join(data_root_3d, split, '*.pth'))

        self.feat_dim = 768

    def __getitem__(self, item):
        data_path = self.data_paths[item]
        locs_in = torch.load(data_path)[0]
        # obtain all camera views related information (specificially for Matterport)
        intrinsics, poses, img_dirs, scene_id, num_img = get_matterport_camera_data(
            data_path, self.data_root_2d, locs_in, self.split)
        keep_features_in_memory = args.keep_features_in_memory

        # load 3D data (point cloud, color and the corresponding labels)
        locs_in = torch.load(data_path)[0]
        n_points = locs_in.shape[0]
        if num_img == 0:
            print('no views inside {}'.format(scene_id))
            return 1

        # extract image features and keep them in the memory
        # default: False (extract image on the fly)
        if keep_features_in_memory and openseg_model is not None:
            img_features = []
            for img_dir in tqdm(img_dirs):
                img_features.append(extract_vlm_feature(img_dir, openseg_model, text_emb))

        n_points_cur = n_points
        counter = torch.zeros((n_points_cur, 1), device=self.device)
        sum_features = torch.zeros((n_points_cur, self.feat_dim), device=self.device)
        ################ Feature Fusion ###################
        vis_id = torch.zeros((n_points_cur, num_img), dtype=int, device=self.device)
        for img_id, img_dir in enumerate(tqdm(img_dirs)):
            # load pose
            pose = poses[img_id]

            # load per-image intrinsic
            intr = intrinsics[img_id]

            # load depth and convert to meter
            depth_dir = img_dir.replace('color', 'depth')
            _, img_type, yaw_id = img_dir.split('/')[-1].split('_')
            depth_dir = depth_dir[:-8] + 'd' + img_type[1] + '_' + yaw_id[0] + '.png'
            depth = imageio.v2.imread(depth_dir) / self.depth_scale

            # calculate the 3d-2d mapping based on the depth
            mapping = np.ones([n_points, 4], dtype=int)
            mapping[:, 1:4] = self.point2img_mapper.compute_mapping(pose, locs_in, depth, intr)
            if mapping[:, 3].sum() == 0:  # no points corresponds to this image, skip
                continue
            mapping = torch.from_numpy(mapping).to(self.device)
            mask = mapping[:, 3]
            vis_id[:, img_id] = mask
            feat_2d = img_features[img_id].to(self.device)

            feat_2d_3d = feat_2d[:, mapping[:, 1], mapping[:, 2]].permute(1, 0)

            counter[mask != 0] += 1
            sum_features[mask != 0] += feat_2d_3d[mask != 0]

        counter[counter == 0] = 1e-5
        feat_bank = sum_features / counter
        point_ids = torch.unique(vis_id.nonzero(as_tuple=False)[:, 0])


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


def main(args):
    seed = 1457
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    #### Dataset specific parameters #####
    img_dim = (640, 512)
    depth_scale = 4000.0
    #######################################
    visibility_threshold = 0.02  # threshold for the visibility check

    args.depth_scale = depth_scale
    args.cut_num_pixel_boundary = 10  # do not use the features on the image boundary
    args.keep_features_in_memory = False  # keep image features in the memory, very expensive
    args.feat_dim = 768  # CLIP feature dimension
    split = args.split
    data_dir = args.data_dir

    data_root = join(data_dir, 'matterport_3d')
    data_root_2d = join(data_dir, 'matterport_2d')
    args.data_root_2d = data_root_2d
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    process_id_range = args.process_id_range

    if split == 'train':  # for training set, export a chunk of point cloud
        args.n_split_points = 20000
        args.num_rand_file_per_scene = 5
    else:  # for the validation set, export the entire point cloud instead of chunks
        args.n_split_points = 2000000
        args.num_rand_file_per_scene = 1

    # load the openseg model
    saved_model_path = args.openseg_model
    args.text_emb = None
    if args.openseg_model != '':
        args.openseg_model = tf2.saved_model.load(saved_model_path,
                                                  tags=[tf2.saved_model.SERVING], )
        args.text_emb = tf2.zeros([1, 1, args.feat_dim])
    else:
        args.openseg_model = None

    # calculate image pixel-3D points correspondances
    args.point2img_mapper = PointCloudToImageMapper(
        image_dim=img_dim,
        visibility_threshold=visibility_threshold,
        cut_bound=args.cut_num_pixel_boundary)

    data_paths = sorted(glob(join(data_root, split, '*.pth')))
    total_num = len(data_paths)

    id_range = None
    if process_id_range is not None:
        id_range = [int(process_id_range[0].split(',')[0]), int(process_id_range[0].split(',')[1])]

    for i in trange(total_num):
        if id_range is not None and \
                (i < id_range[0] or i > id_range[1]):
            print('skip ', i, data_paths[i])
            continue

        process_one_scene(data_paths[i], out_dir, args)


if __name__ == "__main__":
    # args = get_args()
    print("Arguments:")
    # print(args)
    #
    # main(args)
## python matterport_openseg.py --data_dir ../../data --output_dir ../../data/matterport_multiview_openseg --openseg_model ~/workspace/openseg_exported_clip --split train
