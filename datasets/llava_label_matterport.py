import argparse
import copy
import os
import random
import re
import sys
from glob import glob
from os.path import join

import numpy as np
import torch
from tqdm import trange

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '...'))
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
sys.path.append('../libs')
sys.path.append('../llava')
sys.path.append('..')

from llava.constants import DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_PLACEHOLDER
from llava.conversation import conv_templates
from llava.vlm_utils import load_llava_model, load_images, generate_category_from_llava


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


def get_matterport_camera_data(data_path, locs_in, data_root_2d, split):
    '''Get all camera view related information of Matterport3D.'''

    # find bounding box of the current region
    bbox_l = locs_in.min(axis=0)
    bbox_h = locs_in.max(axis=0)

    building_name = data_path.split('/')[-1].split('_')[0]
    scene_id = data_path.split('/')[-1].split('.')[0]
    scene = os.path.join(data_root_2d, split, building_name)
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


def get_args():
    '''Command line arguments.'''

    parser = argparse.ArgumentParser(
        description='Multi-view feature fusion of OpenSeg on Matterport3D.')
    parser.add_argument('--data_dir', default='/storage2/TEV/datasets/Matterport', type=str,
                        help='Where is the base logging directory')
    parser.add_argument('--output_dir', default='', type=str, help='Where is the base logging directory')
    parser.add_argument('--split', type=str, default='test', help='split: "train"| "val" | "test" ')
    parser.add_argument('--openseg_model', type=str, default='', help='Where is the exported OpenSeg model')
    parser.add_argument('--process_id_range', nargs='+', default=None, help='the id range to process')

    # Hyper parameters
    parser.add_argument('--hparams', default=[], nargs="+")
    args = parser.parse_args()
    return args


def process_one_scene(data_3d_path, data_2d_root, split, tokenizer, model, image_processor, conv):
    # obtain all camera views related information (specificially for Matterport)
    # load 3D data (point cloud, color and the corresponding labels)
    locs_in = torch.load(data_3d_path)[0]
    intrinsics, poses, img_dirs, scene_id, num_img = get_matterport_camera_data(data_3d_path, locs_in,
                                                                                data_2d_root, split)
    if num_img == 0:
        print('no views inside {}'.format(scene_id))
        return 1
    k = min(len(img_dirs), 48)
    img_dirs = random.sample(img_dirs, k)
    # images = load_images(img_dirs, (320, 240))
    cls_names = generate_category_from_llava(img_dirs, tokenizer, model, image_processor, conv, image_size=[320, 240])
    output_file = os.path.join(data_2d_root, split, f'{scene_id}_llava.txt')
    print('Finish one scene ...')
    with open(output_file, "w") as file:
        for item in cls_names:
            file.write(item + "\n")


def main(args):
    seed = 1457
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    #######################################
    visibility_threshold = 0.02  # threshold for the visibility check
    split = args.split
    data_dir = args.data_dir

    data_root = join(data_dir, 'matterport_3d_160')
    data_root_2d = join(data_dir, 'matterport_2d')
    process_id_range = args.process_id_range

    data_paths = sorted(glob(join(data_root, split, '*.pth')))
    total_num = len(data_paths)

    id_range = None
    if process_id_range is not None:
        id_range = [int(process_id_range[0].split(',')[0]), int(process_id_range[0].split(',')[1])]
    '''Process one scene.'''
    tokenizer, model, image_processor, conv_mode = load_llava_model(model_path="liuhaotian/llava-v1.6-mistral-7b")
    qs = 'What objects are within the image? Please reply only contains the names of the objects.'
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
    conv = conv_templates[copy.deepcopy(conv_mode)].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)

    for i in trange(total_num):
        if id_range is not None and (i < id_range[0] or i > id_range[1]):
            print('skip ', i, data_paths[i])
            continue

        process_one_scene(data_paths[i], data_root_2d, split, tokenizer, model, image_processor, conv)


if __name__ == "__main__":
    args = get_args()
    print("Arguments:")
    print(args)

    main(args)
# python matterport_openseg.py --data_dir ../../data --output_dir ../../data/matterport_multiview_openseg
# --openseg_model ~/workspace/openseg_exported_clip --split train
