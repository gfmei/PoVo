import numpy as np
import torch
# import torch_scatter




class PointCloudToImageMapper(object):
    def __init__(
            self, image_dim, visibility_threshold=0.1, cut_bound=0, intrinsics=None, device="cpu",
            use_torch=False, eps=1e-6):

        self.image_dim = image_dim
        self.vis_thres = visibility_threshold
        self.cut_bound = cut_bound
        self.intrinsics = intrinsics
        self.eps = eps

        self.device = device
        if use_torch and self.intrinsics is not None:
            self.intrinsics = torch.as_tensor(self.intrinsics, device=device)

    def compute_mapping_torch(self, camera_to_world, coords, depth=None, intrinsic=None, vis_thresh=None):
        """
        :param camera_to_world: 4 x 4
        :param coords: N x 3 format
        :param depth: H x W format
        :param intrinsic: 3x3 format
        :return: mapping, N x 3 format, (H,W,mask)
        """
        device = coords.device
        visibility_threshold = self.vis_thres if vis_thresh is None else vis_thresh
        intrinsic = self.intrinsics if intrinsic is None else intrinsic
        if intrinsic is None:
            raise ValueError("Camera intrinsics are required for projection.")
        intrinsic = torch.as_tensor(intrinsic, device=device)
        camera_to_world = torch.as_tensor(camera_to_world, device=device).float()

        mapping = torch.zeros((3, coords.shape[0]), dtype=torch.long, device=device)
        coords_new = torch.cat([coords, torch.ones([coords.shape[0], 1], dtype=torch.float, device=device)], dim=1).T

        assert coords_new.shape[0] == 4, "[!] Shape error"
        device = camera_to_world.device
        # Create a regularization term (epsilon * Identity matrix)
        try:
            world_to_camera = torch.linalg.pinv(camera_to_world)
        except torch.linalg.LinAlgError as e:
            print(f"Direct inversion failed: {e}")
            print("Applying sign-aware regularization to the diagonal.")

            # Regularize the diagonal based on sign
            diag = torch.diagonal(camera_to_world)
            sign = torch.sign(diag)
            sign[sign == 0] = 1  # Assign positive sign to zero diagonals
            regularization = torch.eye(camera_to_world.size(0), device=camera_to_world.device) * (self.eps * sign)
            camera_to_world_reg = camera_to_world + regularization
            world_to_camera = torch.linalg.inv(camera_to_world_reg)

        p = world_to_camera.float() @ coords_new.float()
        camera_depth = p[2].clone()
        p[2][torch.abs(p[2]) < self.eps] = self.eps
        p[0] = (p[0] * intrinsic[0][0]) / p[2] + intrinsic[0][2]
        p[1] = (p[1] * intrinsic[1][1]) / p[2] + intrinsic[1][2]
        pi = torch.round(p).long()  # simply round the projected coordinates
        inside_mask = (
                torch.isfinite(camera_depth)
                * (camera_depth > self.eps)
                * (pi[0] >= self.cut_bound)
                * (pi[1] >= self.cut_bound)
                * (pi[0] < self.image_dim[0] - self.cut_bound)
                * (pi[1] < self.image_dim[1] - self.cut_bound)
        )
        if depth is not None:
            depth = torch.as_tensor(depth, device=device)
            # print(inside_mask.shape, depth.shape, pi.shape, pi[1][inside_mask].max(), pi[0][inside_mask].max())
            depth_cur = depth[pi[1][inside_mask], pi[0][inside_mask]]
            occlusion_mask = (depth_cur > 0) & (torch.abs(
                depth_cur - camera_depth[inside_mask]) <= visibility_threshold)
            inside_mask[inside_mask == True] = occlusion_mask.clone()
        else:
            front_mask = p[2] > 0  # make sure the depth is in front
            inside_mask = front_mask * inside_mask

        # Keep the nearest surface when several points project to the same pixel.
        if inside_mask.any():
            pixel_ids = pi[1][inside_mask] * self.image_dim[0] + pi[0][inside_mask]
            _, inverse = torch.unique(pixel_ids, return_inverse=True)
            visible_depth = camera_depth[inside_mask]
            min_depth = torch.full((int(inverse.max()) + 1,), float('inf'), device=device)
            min_depth.scatter_reduce_(0, inverse, visible_depth, reduce='amin', include_self=True)
            inside_mask[inside_mask.clone()] = visible_depth - min_depth[inverse] <= 0.2
        new_inside_mask = inside_mask

        mapping[0][new_inside_mask] = pi[1][new_inside_mask]
        mapping[1][new_inside_mask] = pi[0][new_inside_mask]
        mapping[2][new_inside_mask] = 1

        return mapping.T

    def compute_mapping(self, camera_to_world, coords, depth=None, intrinsic=None):
        """
        :param camera_to_world: 4 x 4
        :param coords: N x 3 format
        :param depth: H x W format
        :param intrinsic: 3x3 format
        :return: mapping, N x 3 format, (H,W,mask)
        """
        intrinsic = self.intrinsics if intrinsic is None else intrinsic
        if intrinsic is None:
            raise ValueError("Camera intrinsics are required for projection.")

        mapping = np.zeros((3, coords.shape[0]), dtype=int)
        coords_new = np.concatenate([coords, np.ones([coords.shape[0], 1])], axis=1).T
        assert coords_new.shape[0] == 4, "[!] Shape error"

        world_to_camera = np.linalg.inv(camera_to_world)
        p = np.matmul(world_to_camera, coords_new)
        camera_depth = p[2].copy()
        p[2][np.abs(p[2]) < self.eps] = self.eps
        p[0] = (p[0] * intrinsic[0][0]) / p[2] + intrinsic[0][2]
        p[1] = (p[1] * intrinsic[1][1]) / p[2] + intrinsic[1][2]
        pi = np.round(p).astype(np.int32)  # simply round the projected coordinates
        inside_mask = (
                np.isfinite(camera_depth)
                * (camera_depth > self.eps)
                * (pi[0] >= self.cut_bound)
                * (pi[1] >= self.cut_bound)
                * (pi[0] < self.image_dim[0] - self.cut_bound)
                * (pi[1] < self.image_dim[1] - self.cut_bound)
        )
        if depth is not None:
            depth_cur = depth[pi[1][inside_mask], pi[0][inside_mask]]
            occlusion_mask = (
                    (depth_cur > 0) & (np.abs(depth_cur - camera_depth[inside_mask]) <= self.vis_thres)
            )
            inside_mask[inside_mask == True] = occlusion_mask
        else:
            front_mask = p[2] > 0  # make sure the depth is in front
            inside_mask = front_mask * inside_mask

        # NOTE detect occlusion
        pi_x_ = pi[1][inside_mask]
        pi_y_ = pi[0][inside_mask]
        pi_depth_ = camera_depth[inside_mask]

        if not inside_mask.any():
            return mapping.T

        inds = (pi_x_ * self.image_dim[0] + pi_y_).astype(np.int32)
        _, inds = np.unique(inds, return_inverse=True)

        depth_min = np.full(inds.max() + 1, np.inf)
        np.minimum.at(depth_min, inds, pi_depth_)
        depth_min_broadcast = depth_min[inds]

        THRESHOLD = 0.2  # (meter)
        depth_occlusion_mask = (pi_depth_ - depth_min_broadcast) <= THRESHOLD

        new_inside_mask = inside_mask.copy()
        new_inside_mask[inside_mask] = depth_occlusion_mask
        ############################

        mapping[0][new_inside_mask] = pi[1][new_inside_mask]
        mapping[1][new_inside_mask] = pi[0][new_inside_mask]
        mapping[2][new_inside_mask] = 1

        return mapping.T
