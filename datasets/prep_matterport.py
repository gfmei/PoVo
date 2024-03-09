import glob
import multiprocessing as mp
import os

import cv2
import imageio
import numpy as np
import pandas as pd
import plyfile
import torch
from tqdm import tqdm

from data_util import adjust_intrinsic


def remove_items(test_list, item):
    return [i for i in test_list if i != item]


def obtain_intr_extr_matterport(file):
    """Obtain the intrinsic and extrinsic parameters of Matterport3D."""
    lines = file.readlines()
    intrinsics = []
    extrinsics = []
    img_names = []
    for i, line in enumerate(lines):
        line = line.strip()
        if 'intrinsics_matrix' in line:
            line = line.replace('intrinsics_matrix ', '')
            line = line.split(' ')
            line = remove_items(line, '')
            if len(line) != 9:
                print('something wrong at {}'.format(i))
            intrinsic = np.asarray(line).astype(float).reshape(3, 3)
            intrinsics.extend([intrinsic, intrinsic, intrinsic, intrinsic, intrinsic, intrinsic])
        elif 'scan' in line:
            line = line.split(' ')
            img_names.append(line[2])

            line = remove_items(line, '')[3:]
            if len(line) != 16:
                print('something wrong at {}'.format(i))
            extrinsic = np.asarray(line).astype(float).reshape(4, 4)
            extrinsics.append(extrinsic)

    intrinsics = np.stack(intrinsics, axis=0)
    extrinsics = np.stack(extrinsics, axis=0)
    img_names = np.asarray(img_names)

    return img_names, intrinsics, extrinsics


def process_txt(filename):
    with open(filename) as file:
        lines = file.readlines()
        lines = [line.rstrip() for line in lines]
    return lines


def label_mapping(num_classes=160, tsv_file_root=None):
    #####################################
    if tsv_file_root is None:
        tsv_file_root = './meta_data/category_mapping.tsv'
    category_mapping = pd.read_csv(tsv_file_root, sep='\t', header=0)
    # obtain label mapping for new number of classes
    label_name = []
    label_id = []
    label_all = category_mapping['nyuClass'].tolist()
    eliminated_list = ['void', 'unknown']
    mapping = np.zeros(len(label_all) + 1, dtype=int)  # mapping from category id
    instance_count = category_mapping['count'].tolist()
    ins_count_list = []
    counter = 1
    flag_stop = False
    for i, x in enumerate(label_all):
        if not flag_stop and isinstance(x, str) and x not in label_name and x not in eliminated_list:
            label_name.append(x)
            label_id.append(counter)
            mapping[i + 1] = counter
            counter += 1
            ins_count_list.append(instance_count[i])
            if counter == num_classes + 1:
                flag_stop = True
        elif isinstance(x, str) and x in label_name:
            # find the index of the previously appeared object name
            mapping[i + 1] = label_name.index(x) + 1

    return mapping, label_all, label_name


def process_one_scene_from_pcd(fn, num_classes, mapping, out_dir):
    '''process one scene.'''
    scene_name = fn.split('/')[-3]
    region_name = fn.split('/')[-1].split('.')[0]
    a = plyfile.PlyData().read(fn)
    v = np.array([list(x) for x in a.elements[0]])
    coords = np.ascontiguousarray(v[:, :3])
    colors = np.ascontiguousarray(v[:, -3:]) / 127.5 - 1

    category_id = a['face']['category_id']
    category_id[category_id == -1] = 0
    mapped_labels = mapping[category_id]
    triangles = a['face']['vertex_indices']
    vertex_labels = np.zeros((coords.shape[0], num_classes + 1), dtype=np.int32)
    # calculate per-vertex labels
    for row_id in range(triangles.shape[0]):
        for i in range(3):
            vertex_labels[triangles[row_id][i], mapped_labels[row_id]] += 1

    vertex_labels = np.argmax(vertex_labels, axis=1)
    vertex_labels[vertex_labels == 0] = 256
    vertex_labels -= 1
    torch.save((coords, colors, vertex_labels), os.path.join(out_dir, scene_name + '_' + region_name + '.pth'))
    print('Preprocessing file: {}'.format(fn))


def process_one_scene_from_images(fn, scene, in_path, out_dir, img_dim, orig_img_dim):
    '''process one scene.'''
    out_dir_color = os.path.join(out_dir, scene, 'color')
    out_dir_depth = os.path.join(out_dir, scene, 'depth')
    out_dir_pose = os.path.join(out_dir, scene, 'pose')
    out_dir_intrinsic = os.path.join(out_dir, scene, 'intrinsic')
    if not os.path.exists(out_dir_color):
        os.makedirs(out_dir_color)
    if not os.path.exists(out_dir_depth):
        os.makedirs(out_dir_depth)
    if not os.path.exists(out_dir_pose):
        os.makedirs(out_dir_pose)
    if not os.path.exists(out_dir_intrinsic):
        os.makedirs(out_dir_intrinsic)
    # save the camera parameters to the folder
    camera_dir = os.path.join(in_path, scene, scene, 'undistorted_camera_parameters', '{}.conf'.format(scene))
    img_names, intr_list, pose_list = obtain_intr_extr_matterport(open(camera_dir))

    # process RGB images
    img_name = fn.split('/')[-1]
    img_id = np.where(img_names == img_name)[0].item()

    img = imageio.v3.imread(fn)
    img = cv2.resize(img, img_dim, interpolation=cv2.INTER_NEAREST)
    imageio.imwrite(os.path.join(out_dir_color, img_name), img)

    # process depth images
    pano_d, img_type, yaw_id = fn.split('/')[-1].split('_')
    fn_depth = fn.replace('color', 'depth')
    fn_depth = fn_depth[:-8] + 'd' + img_type[1] + '_' + yaw_id[0] + '.png'
    depth_name = fn_depth.split('/')[-1]
    depth = imageio.v3.imread(fn_depth).astype(np.uint16)
    depth = cv2.resize(depth, img_dim, interpolation=cv2.INTER_NEAREST)
    imageio.imwrite(os.path.join(out_dir_depth, depth_name), depth)

    # process poses
    file_name = img_name.split('.jpg')[0]
    pose = pose_list[img_id]
    pose[:3, 1] *= -1.0
    pose[:3, 2] *= -1.0
    np.savetxt(os.path.join(out_dir_pose, file_name + '.txt'), pose)

    # process intrinsic parameters
    intrinsic = adjust_intrinsic(intr_list[img_id], orig_img_dim, img_dim)
    np.savetxt(os.path.join(out_dir_intrinsic, file_name + '.txt'), intrinsic)


# ! YOU NEED TO MODIFY THE FOLLOWING
#####################################
def generate_2d_data(split='test'):
    # split: 'train' | 'val' | 'test'
    out_dir = '/storage2/TEV/datasets/Matterport/matterport_2d/'
    in_path = '/storage2/TEV/datasets/Matterport/test_raw'  # downloaded original matterport data
    scene_list = process_txt('meta_data/matterport/scenes_{}.txt'.format(split))
    #####################################
    os.makedirs(out_dir, exist_ok=True)
    # Meta Data #######
    img_dim = (640, 512)
    original_img_dim = (1280, 1024)

    for scene in tqdm(scene_list):
        files = glob.glob(os.path.join(in_path, scene, scene, 'undistorted_color_images', '*.jpg'))
        # Use multiprocessing to process files in parallel
        with mp.Pool(processes=mp.cpu_count()) as p:
            p.map(process_one_scene_from_images, files, scene, in_path, out_dir, img_dim, original_img_dim)


def generate_3d_data(num_classes=160, split='test'):
    # ! YOU NEED TO MODIFY THE FOLLOWING
    """
    :param num_classes: 40 | 80 | 160 # define the number of classes
    :param split: 'train' | 'val' | 'test'
    :return:
    """
    #####################################
    out_dir = '/storage2/TEV/datasets/Matterport/matterport_3d_{}/{}'.format(num_classes, split)
    matterport_path = '/storage2/TEV/datasets/Matterport/test_raw'  # downloaded original matterport data
    scene_list = process_txt('./meta_data/scenes_{}.txt'.format(split))
    #####################################
    os.makedirs(out_dir, exist_ok=True)
    # obtain label mapping for new number of classes
    mapping, label_all, label_name = label_mapping(num_classes)
    output_file = '/storage2/TEV/datasets/Matterport/matterport_3d_{}/{}_class_name.txt'.format(num_classes, split)
    with open(output_file, "w") as file:
        for item in label_name:
            file.write(item + "\n")
    files = []
    for scene in scene_list:
        files = files + glob.glob(os.path.join(matterport_path, scene, scene, 'region_segmentations', '*.ply'))
    args_list = [(fn, num_classes, mapping, out_dir) for fn in files]
    with mp.Pool(processes=mp.cpu_count()) as p:
        p.starmap(process_one_scene_from_pcd, args_list)


if __name__ == '__main__':
    generate_3d_data(160, 'test')
