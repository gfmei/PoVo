import os
from os.path import join, dirname, basename, splitext
from typing import Any, Dict

import cv2
import numpy as np

import torch
from PIL import Image
from glob import glob

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
        pose_path = join(dirname(dirname(img_dir)), 'pose', splitext(basename(img_dir))[0] + '.txt')
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
            tuple: (points, spp, ins_gt, sem_gt) where:
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
        pcd_loader = torch.load(data_path, map_location=self.device, weights_only=False)
        points = torch.as_tensor(pcd_loader['coords'], dtype=torch.float32, device=self.device)
        ins_gt = torch.as_tensor(pcd_loader['instance_ids'], dtype=torch.long, device=self.device)
        spp = torch.as_tensor(pcd_loader['spts'], dtype=torch.long, device=self.device)
        _, spp = torch.unique(spp, return_inverse=True)
        sem_gt = torch.as_tensor(pcd_loader['labels'], dtype=torch.long, device=self.device)
        return points, spp, ins_gt, sem_gt

    def get_point_colors(self, scene_id, split):
        """Read mesh RGB, converting the prepared [-1, 1] colors to [0, 1]."""
        path = join(self.root_path, 'scannet200', split, f'{scene_id}.pth')
        data = torch.load(path, map_location='cpu', weights_only=False)
        if 'colors' not in data:
            return None
        colors = torch.as_tensor(data['colors'], dtype=torch.float32)
        if colors.shape != (len(data['coords']), 3):
            raise ValueError('cached mesh colors must have one RGB vector per point')
        finite = colors[torch.isfinite(colors).all(1)]
        if finite.numel() and ((finite < -1).any() or (finite > 1).any()):
            raise ValueError('prepared mesh colors must lie between -1 and 1')
        return ((colors + 1) / 2).to(self.device)

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
        depth_path = join(dirname(dirname(img_dir)), 'depth', splitext(basename(img_dir))[0] + '.png')
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


    def get_grounded_data(self, img_dir, split, scene_id, name='sam2'):
        """Load cached masks, CLIP features, and confidences saved by save_mask."""
        color_name = splitext(basename(img_dir))[0]
        relative_path = join(split, scene_id, name, f'{color_name}.pth')
        for folder in ('masks', 'mask_sam2'):
            mask_path = join(self.root_path, folder, relative_path)
            if os.path.exists(mask_path):
                return torch.load(mask_path, map_location=self.device, weights_only=False)
        raise FileNotFoundError(f"Grounded mask data not found for frame {color_name} of {scene_id}.")

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
        data_root_mask = join(self.root_path, 'masks', split, scene_id, 'sam2')

        # Extract the base name of the image file without extension
        color_name = splitext(basename(img_path))[0]

        # Define the full path for the mask file
        mask_path = join(data_root_mask, f'{color_name}.pth')

        # Create the directory if it doesn't exist
        os.makedirs(data_root_mask, exist_ok=True)

        # Save the segmentation masks using torch.save
        torch.save(grounded_data_dict, mask_path)


class SelectedScanReader(ScanReader):
    """Read the frame IDs selected in a prepared cache manifest."""
    selected = None

    def get_image_dirs(self, scene_id, split):
        return [path for path in super().get_image_dirs(scene_id, split)
                if self.selected is None or int(splitext(basename(path))[0]) in self.selected]
