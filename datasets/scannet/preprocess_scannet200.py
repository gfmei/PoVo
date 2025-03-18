import argparse
import glob
import json
import os, sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
import multiprocessing as mp
import numpy as np
import pandas as pd
import torch
from plyfile import PlyData, PlyElement

current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
# Add directories to sys.path
sys.path.append(current_dir)
sys.path.append(parent_dir)

from datasets.meta_data.scannet200_constants import VALID_CLASS_IDS_200


def face_normal(vertex, face):
    v01 = vertex[face[:, 1]] - vertex[face[:, 0]]
    v02 = vertex[face[:, 2]] - vertex[face[:, 0]]
    vec = np.cross(v01, v02)
    length = np.sqrt(np.sum(vec ** 2, axis=1, keepdims=True)) + 1.0e-8
    nf = vec / length
    area = length * 0.5
    return nf, area


def vertex_normal(vertex, face):
    nf, area = face_normal(vertex, face)
    nf = nf * area

    nv = np.zeros_like(vertex)
    for i in range(face.shape[0]):
        nv[face[i]] += nf[i]

    length = np.sqrt(np.sum(nv ** 2, axis=1, keepdims=True)) + 1.0e-8
    nv = nv / length
    return nv


# def read_plymesh(filepath):
#     """Read ply file and return it as numpy array. Returns None if emtpy."""
#     with open(filepath, 'rb') as f:
#         plydata = plyfile.PlyData.read(f)
#     if plydata.elements:
#         vertices = pd.DataFrame(plydata['vertex'].data).values
#         faces = np.stack(plydata['face'].data['vertex_indices'], axis=0)
#         return vertices, faces


def read_plymesh(filepath):
    """Read ply file and return it as numpy array. Returns None if emtpy."""
    with open(filepath, "rb") as f:
        plydata = PlyData.read(f)
    if plydata.elements:
        vertices = pd.DataFrame(plydata["vertex"].data).values
        faces = np.stack(plydata['face'].data['vertex_indices'], axis=0)
        # faces = np.array([f[0] for f in plydata["face"].data])
        return vertices, faces
    else:
        raise EOFError


def save_plymesh(vertices, faces, filename, verbose=True, with_label=True):
    """Save an RGB point cloud as a PLY file.

    Args:
      points_3d: Nx6 matrix where points_3d[:, :3] are the XYZ coordinates and points_3d[:, 4:] are
          the RGB values. If Nx3 matrix, save all points with [128, 128, 128] (gray) color.
    """
    assert vertices.ndim == 2
    if with_label:
        if vertices.shape[1] == 7:
            python_types = (float, float, float, int, int, int, int)
            npy_types = [
                ("x", "f4"),
                ("y", "f4"),
                ("z", "f4"),
                ("red", "u1"),
                ("green", "u1"),
                ("blue", "u1"),
                ("label", "u4"),
            ]

        if vertices.shape[1] == 8:
            python_types = (float, float, float, int, int, int, int, int)
            npy_types = [
                ("x", "f4"),
                ("y", "f4"),
                ("z", "f4"),
                ("red", "u1"),
                ("green", "u1"),
                ("blue", "u1"),
                ("label", "u4"),
                ("instance_id", "u4"),
            ]

    else:
        if vertices.shape[1] == 3:
            gray_concat = np.tile(np.array([128], dtype=np.uint8), (vertices.shape[0], 3))
            vertices = np.hstack((vertices, gray_concat))
        elif vertices.shape[1] == 6:
            python_types = (float, float, float, int, int, int)
            npy_types = [("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")]
        else:
            pass

    vertices_list = []
    for row_idx in range(vertices.shape[0]):
        cur_point = vertices[row_idx]
        vertices_list.append(tuple(dtype(point) for dtype, point in zip(python_types, cur_point)))
    vertices_array = np.array(vertices_list, dtype=npy_types)
    elements = [PlyElement.describe(vertices_array, "vertex")]

    if faces is not None:
        faces_array = np.empty(len(faces), dtype=[("vertex_indices", "i4", (3,))])
        faces_array["vertex_indices"] = faces
        elements += [PlyElement.describe(faces_array, "face")]

    # Write
    PlyData(elements).write(filename)

    if verbose is True:
        print("Saved point cloud to: %s" % filename)


# Map the raw category id to the point cloud
def point_indices_from_group(points, seg_indices, group, labels_pd, CLASS_IDs):
    group_segments = np.array(group["segments"])
    label = group["label"]

    # Map the category name to id
    label_ids = labels_pd[labels_pd["raw_category"] == label]["id"]
    label_id = int(label_ids.iloc[0]) if len(label_ids) > 0 else 0

    # Only store for the valid categories
    if label_id not in CLASS_IDs:
        label_id = 0

    # get points, where segment indices (points labelled with segment ids) are in the group segment list
    point_IDs = np.where(np.isin(seg_indices, group_segments))

    return points[point_IDs], point_IDs[0], label_id


warnings.filterwarnings("ignore", category=DeprecationWarning)
IGNORE_INDEX = -1

CLOUD_FILE_PFIX = "_vh_clean_2"
SEGMENTS_FILE_PFIX = ".0.010000.segs.json"
AGGREGATIONS_FILE_PFIX = ".aggregation.json"
CLASS_IDs = VALID_CLASS_IDS_200

NORMALIZED_CKASS_IDS_200 = [-100 for _ in range(1192)]
REVERSE_NORMALIZED_CKASS_IDS_200 = [-100 for _ in range(200)]

count_id = 2
for i, cls_id in enumerate(VALID_CLASS_IDS_200):
    if cls_id == 1:  # wall
        NORMALIZED_CKASS_IDS_200[cls_id] = 0
        REVERSE_NORMALIZED_CKASS_IDS_200[0] = cls_id
    elif cls_id == 3:  # floor
        NORMALIZED_CKASS_IDS_200[cls_id] = 1
        REVERSE_NORMALIZED_CKASS_IDS_200[1] = cls_id
    else:
        NORMALIZED_CKASS_IDS_200[cls_id] = count_id
        REVERSE_NORMALIZED_CKASS_IDS_200[count_id] = cls_id
        count_id += 1

REVERSE_NORMALIZED_CKASS_IDS_200_np = np.array(REVERSE_NORMALIZED_CKASS_IDS_200)
np.save("reverse_norm_ids.npy", REVERSE_NORMALIZED_CKASS_IDS_200_np)



def handle_process(scene_path, output_path, labels_pd, train_scenes, val_scenes):
    scene_id = scene_path.split("/")[-1]
    mesh_path = os.path.join(scene_path, f"{scene_id}{CLOUD_FILE_PFIX}.ply")
    segments_file = os.path.join(scene_path, f"{scene_id}{CLOUD_FILE_PFIX}{SEGMENTS_FILE_PFIX}")
    aggregations_file = os.path.join(scene_path, f"{scene_id}{AGGREGATIONS_FILE_PFIX}")
    info_file = os.path.join(scene_path, f"{scene_id}.txt")

    if scene_id in train_scenes:
        output_file = os.path.join(output_path, "train", f"{scene_id}.pth")
        split_name = "train"
    elif scene_id in val_scenes:
        output_file = os.path.join(output_path, "val", f"{scene_id}.pth")
        split_name = "val"
    else:
        output_file = os.path.join(output_path, "test", f"{scene_id}.pth")
        split_name = "test"

    print("Processing: ", scene_id, "in ", split_name)

    # Rotating the mesh to axis aligned
    info_dict = {}
    with open(info_file) as f:
        for line in f:
            (key, val) = line.split(" = ")
            info_dict[key] = np.fromstring(val, sep=" ")

    # if "axisAlignment" not in info_dict:
    #     rot_matrix = np.identity(4)
    # else:
    #     rot_matrix = info_dict["axisAlignment"].reshape(4, 4)

    pointcloud, faces_array = read_plymesh(mesh_path)
    points = pointcloud[:, :3]
    colors = pointcloud[:, 3:6]
    # alphas = pointcloud[:, -1]

    # Rotate PC to axis aligned
    # r_points = pointcloud[:, :3].transpose() # type: ignore
    # r_points = np.append(r_points, np.ones((1, r_points.shape[1])), axis=0)
    # r_points = np.dot(rot_matrix, r_points)
    # pointcloud = np.append(r_points.transpose()[:, :3], pointcloud[:, 3:], axis=1) # type: ignore

    # Load segments file
    with open(segments_file) as f:
        segments = json.load(f)
        seg_indices = np.array(segments["segIndices"])

    # Load Aggregations file
    with open(aggregations_file) as f:
        aggregation = json.load(f)
        seg_groups = np.array(aggregation["segGroups"])

    # Generate new labels
    labelled_pc = np.zeros((pointcloud.shape[0], 1))
    instance_ids = np.zeros((pointcloud.shape[0], 1))
    for group in seg_groups:
        segment_points, p_inds, label_id = point_indices_from_group(
            pointcloud, seg_indices, group, labels_pd, CLASS_IDs
        )

        # labelled_pc[p_inds] = label_id
        labelled_pc[p_inds] = NORMALIZED_CKASS_IDS_200[label_id]
        instance_ids[p_inds] = group["id"]

    labelled_pc = labelled_pc.astype(int)
    instance_ids = instance_ids.astype(int)

    torch.save(
        {
            'coords': points,
            'colors': colors / 127.5 - 1.0,
            'labels': labelled_pc.reshape(-1),
            "normal": vertex_normal(points, faces_array),
            "spts": seg_indices.reshape(-1),
            'instance_ids': instance_ids.reshape(-1)
        },
        output_file,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", default='/data/disk1/data/scannet/scans',
                        help="Path to the ScanNet dataset containing scene folders")
    parser.add_argument("--output_root", default='/data/disk1/data/scannet/scannet200',
                        help="Output path where train/val folders will be located")
    parser.add_argument("--label_map_file", default='datasets/meta_data/scannetv2-labels.combined.tsv',
                        help="path to scannetv2-labels.combined.tsv")
    parser.add_argument("--num_workers", default=16, type=int, help="The number of parallel workers")
    parser.add_argument(
        "--train_val_splits_path",
        default="datasets/meta_data",
        help="Where the txt files with the train/val splits live",
    )
    config = parser.parse_args()

    # Load label map
    labels_pd = pd.read_csv(config.label_map_file, sep="\t", header=0)

    # Load train/val splits
    with open(os.path.join(config.train_val_splits_path, "scannetv2_train.txt")) as train_file:
        train_scenes = train_file.read().splitlines()
    with open(os.path.join(config.train_val_splits_path, "scannetv2_val.txt")) as val_file:
        val_scenes = val_file.read().splitlines()

    # Create output directories
    train_output_dir = os.path.join(config.output_root, "train")
    if not os.path.exists(train_output_dir):
        os.makedirs(train_output_dir)
    val_output_dir = os.path.join(config.output_root, "val")
    if not os.path.exists(val_output_dir):
        os.makedirs(val_output_dir)

    # Load scene paths
    scene_paths = sorted(glob.glob(config.dataset_root + "/*"))
    
    
    with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
        # Load scene paths
        scene_paths = scene_paths

        # Preprocess data.
        print('Processing scenes...')
        # pool = ProcessPoolExecutor(max_workers=mp.cpu_count())
        # pool = ProcessPoolExecutor(max_workers=1)
        _ = list(executor.map(handle_process, scene_paths, repeat(config.output_root), repeat(labels_pd),
                              repeat(train_scenes), repeat(val_scenes)))

    # # Preprocess data.
    # pool = ProcessPoolExecutor(max_workers=config.num_workers)
    # print("Processing scenes...")
    # _ = list(
    #     pool.map(
    #         handle_process,
    #         scene_paths,
    #         repeat(config.output_root),
    #         repeat(labels_pd),
    #         repeat(train_scenes),
    #         repeat(val_scenes),
    #     )
    # )
