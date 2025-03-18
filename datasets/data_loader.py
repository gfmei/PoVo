import os
import sys
from os.path import join, dirname, abspath, basename, splitext
from typing import Any, Dict

import cv2
import numpy as np

import torch
from PIL import Image
from glob2 import glob
# Define directories
current_dir = dirname(__file__)
parent_dir = abspath(join(current_dir, '..'))
libs_dir = abspath(join(parent_dir, 'libs'))
data_dir = abspath(join(parent_dir, 'datasets'))
# Add directories to sys.path
sys.path.extend([current_dir, parent_dir, libs_dir, data_dir])

# Project-specific imports
from libs.lib_data import num_to_natural_torch


class ScanReader:
    """
    A utility class to handle ScanNet data reading for 3D scene understanding tasks.

    Args:
        root_path (str): Path to the root directory of the ScanNet dataset.
        device (str): Device to transfer the data to ('cuda' or 'cpu').
    """

    def __init__(self, root_path, device):
        self.root_path = root_path
        self.device = device

    def get_pose(self, img_dir):
        """
        Retrieve the pose matrix for a given image directory.
        Args:
            img_dir (str): Directory of the input image.
        Returns:
            np.ndarray: Pose matrix as a float numpy array.
        Raises:
            FileNotFoundError: If the pose file is not found.
        """
        pose_path = img_dir.replace('color', 'pose').replace('.jpg', '.txt')
        if not os.path.exists(pose_path):
            raise FileNotFoundError(f"Pose file not found at {pose_path}.")
        pose = np.loadtxt(pose_path, dtype=float)
        return pose

    def get_image_dirs(self, scene_id, split):
        """
        Retrieve sorted list of image directories for a given scene and split.
        Args:
            scene_id (str): Scene identifier.
            split (str): Dataset split ('train', 'val', 'test').

        Returns:
            list: Sorted list of image file paths.
        """
        data_root_2d = join(self.root_path, 'scannet_2d')
        img_pattern = join(data_root_2d, split, scene_id, 'color', '*.jpg')
        img_dirs = sorted(glob(img_pattern), key=lambda x: int(os.path.basename(x)[:-4]))
        return img_dirs

    def get_pcd(self, scene_id, split):
        """
        Load point cloud data for a given scene and split.
        Args:
            scene_id (str): Scene identifier.
            split (str): Dataset split ('train', 'val', 'test').

        Returns:
            tuple: (points, spp, ins_gt) where:
                - points (torch.Tensor): Point cloud coordinates.
                - spp (torch.Tensor): Superpoint labels.
                - ins_gt (torch.Tensor): Instance ground truth labels.
        Raises:
            FileNotFoundError: If the point cloud data file is not found.
        """
        scene_name = f"{scene_id}.pth"
        data_path = join(self.root_path, 'scannet200', split, scene_name)
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Point cloud data not found for scene {scene_id} at {data_path}.")
        pcd_loader = torch.load(data_path, map_location=self.device)
        points = torch.tensor(pcd_loader['coords'], device=self.device)
        ins_gt = torch.tensor(pcd_loader['instance_ids'], device=self.device)
        spp = torch.tensor(pcd_loader['spts'], device=self.device)
        spp = num_to_natural_torch(spp, void_number=0) - 1
        sem_gt = torch.tensor(pcd_loader['labels'], device=self.device)
        return points, spp, ins_gt, sem_gt

    def get_intrinsic(self, scene_id):
        """
        Retrieve intrinsic parameters for a given scene.
        Args:
            scene_id (str): Scene identifier.
        Returns:
            np.ndarray: Intrinsic matrix.
        Raises:
            FileNotFoundError: If the intrinsic file is not found.
        """
        intrinsic_path = join(self.root_path, "intrinsic", f"{scene_id}.txt")
        if not os.path.exists(intrinsic_path):
            return None
        intrinsic = np.loadtxt(intrinsic_path, dtype=float)
        return intrinsic

    def get_depth(self, img_dir):
        """
        Load depth map for a given image directory.
        Args:
            img_dir (str): Directory of the input image.
        Returns:
            np.ndarray: Depth map as a numpy array.
        Raises:
            FileNotFoundError: If the depth map file is not found.
            ValueError: If the depth map is invalid or cannot be read.
        """
        depth_path = img_dir.replace('color', 'depth').replace('.jpg', '.png')
        if not os.path.exists(depth_path):
            raise FileNotFoundError(f"Depth map not found at {depth_path}.")
        depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
        if depth is None:
            raise ValueError(f"Invalid depth map at {depth_path}.")
        return depth

    def get_image(self, img_dir):
        """
        Load an RGB image from the specified directory.
        Args:
            img_dir (str): Directory of the input image.
        Returns:
            PIL.Image.Image: Loaded RGB image.
        """
        return Image.open(img_dir).convert("RGB")

    def get_mask(self, img_dir, split, scene_id):
        """
        Load segmentation masks for a given image directory.
        Args:
            img_dir (str): Directory of the input image.
            split (str): Dataset split ('train', 'val', 'test').
            scene_id (str): Scene identifier.
        Returns:
            torch.Tensor: Segmentation masks.
        """
        data_root_mask = join(self.root_path, 'scannet_mask2d', split, scene_id, 'sam')
        color_name = os.path.splitext(os.path.basename(img_dir))[0]
        mask_path = join(data_root_mask, f'maskraw_{color_name}.pth')
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask file not found at {mask_path}.")
        masks = torch.load(mask_path, map_location=self.device)['mask']
        return masks
    
    def save_mask(self, img_path: str, split: str, scene_id: str, grounded_data_dict: Dict[str, Any]) -> None:
        """
        Save segmentation masks for a given image.

        Args:
            img_path (str): Path to the input image file.
            split (str): Dataset split ('train', 'val', 'test').
            scene_id (str): Scene identifier.
            grounded_data_dict (dict): Segmentation masks data to be saved.

        Returns:
            None
        """
        # Construct the directory path where masks will be saved
        data_root_mask = join(self.root_path, 'mask_sam2', split, scene_id, 'sam2')

        # Extract the base name of the image file without extension
        color_name = splitext(basename(img_path))[0]

        # Define the full path for the mask file
        mask_path = join(data_root_mask, f'{color_name}.pth')

        # Create the directory if it doesn't exist
        os.makedirs(data_root_mask, exist_ok=True)

        # Save the segmentation masks using torch.save
        try:
            torch.save(grounded_data_dict, mask_path)
            print(f"Mask successfully saved to {mask_path}")
        except Exception as e:
            print(f"Error saving mask to {mask_path}: {e}")