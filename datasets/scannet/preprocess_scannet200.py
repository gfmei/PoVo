import argparse
import glob
import json
import os, sys
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
import multiprocessing as mp
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

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


def read_plymesh(filepath):
    """Read ply file and return it as numpy array. Returns None if emtpy."""
    import pandas as pd
    from plyfile import PlyData

    with open(filepath, "rb") as f:
        plydata = PlyData.read(f)
    if plydata.elements:
        vertices = pd.DataFrame(plydata["vertex"].data).values
        faces = np.stack(plydata['face'].data['vertex_indices'], axis=0)
        return vertices, faces
    else:
        raise EOFError


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


CLOUD_FILE_PFIX = "_vh_clean_2"
SEGMENTS_FILE_PFIX = ".0.010000.segs.json"
AGGREGATIONS_FILE_PFIX = ".aggregation.json"
CLASS_IDs = VALID_CLASS_IDS_200

NORMALIZED_CKASS_IDS_200 = [-100 for _ in range(1192)]

count_id = 2
for i, cls_id in enumerate(VALID_CLASS_IDS_200):
    if cls_id == 1:  # wall
        NORMALIZED_CKASS_IDS_200[cls_id] = 0
    elif cls_id == 3:  # floor
        NORMALIZED_CKASS_IDS_200[cls_id] = 1
    else:
        NORMALIZED_CKASS_IDS_200[cls_id] = count_id
        count_id += 1


def handle_process(scene_path, output_path, labels_pd, train_scenes, val_scenes):
    scene_id = scene_path.split("/")[-1]
    mesh_path = os.path.join(scene_path, f"{scene_id}{CLOUD_FILE_PFIX}.ply")
    segments_file = os.path.join(scene_path, f"{scene_id}{CLOUD_FILE_PFIX}{SEGMENTS_FILE_PFIX}")
    aggregations_file = os.path.join(scene_path, f"{scene_id}{AGGREGATIONS_FILE_PFIX}")

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

    pointcloud, faces_array = read_plymesh(mesh_path)
    points = pointcloud[:, :3]
    colors = pointcloud[:, 3:6]

    # Keep mesh coordinates in the sensor world frame used by the RGB-D poses.

    # Load segments file
    with open(segments_file) as f:
        segments = json.load(f)
        seg_indices = np.array(segments["segIndices"])

    # Load Aggregations file
    if split_name == "test" and not os.path.exists(aggregations_file):
        seg_groups = []
    else:
        with open(aggregations_file) as f:
            aggregation = json.load(f)
            seg_groups = aggregation["segGroups"]

    # Generate new labels
    labelled_pc = np.full((pointcloud.shape[0], 1), -100, dtype=np.int64)
    instance_ids = np.full((pointcloud.shape[0], 1), -1, dtype=np.int64)
    for group in seg_groups:
        _, p_inds, label_id = point_indices_from_group(
            pointcloud, seg_indices, group, labels_pd, CLASS_IDs
        )

        labelled_pc[p_inds] = NORMALIZED_CKASS_IDS_200[label_id]
        instance_ids[p_inds] = group["id"]

    labelled_pc = labelled_pc.astype(int)
    instance_ids = instance_ids.astype(int)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
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
    import pandas as pd

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


    with ProcessPoolExecutor(max_workers=config.num_workers) as executor:
        # Load scene paths

        # Preprocess data.
        print('Processing scenes...')
        _ = list(executor.map(handle_process, scene_paths, repeat(config.output_root), repeat(labels_pd),
                              repeat(train_scenes), repeat(val_scenes)))
