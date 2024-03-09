import numpy as np
import open3d as o3d
import torch
import trimesh

try:
    import segmentator
except Exception as e:
    raise ImportError


def to_o3d_pcd(pcd, est_normal=False, radius=0.05):
    '''
    Transfer a point cloud of numpy.ndarray to open3d point cloud
    :param radius:
    :param est_normal:
    :param pcd: point cloud of numpy.ndarray in shape[N, 3]
    :return: open3d.geometry.PointCloud()
    '''
    pcd_ = o3d.geometry.PointCloud()
    pcd_.points = o3d.utility.Vector3dVector(pcd)
    if est_normal:
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=33))
    return pcd_


def generate_mesh_from_pcd(xyz, normal=None):
    pcd = to_o3d_pcd(xyz)
    if normal is None:
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    # estimate radius for rolling ball
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = 1.5 * avg_dist
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd,
        o3d.utility.DoubleVector([radius, radius * 2]))

    # create the triangular mesh with the vertices and faces from open3d
    tri_mesh = trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles),
                               vertex_normals=np.asarray(mesh.vertex_normals))
    # trimesh.convex.is_convex(tri_mesh)
    return tri_mesh


def get_super_point_cloud(mesh, xyz):
    # mesh = trimesh.load_mesh(mesh_file)
    _vertices = torch.from_numpy(mesh.vertices.astype(np.float32))
    _faces = torch.from_numpy(mesh.faces.astype(np.int64))
    superpoint = segmentator.segment_mesh(_vertices, _faces).numpy()
    # creat_labeled_point_cloud(xyz, superpoint, 'label_spts')
    spnum = len(np.unique(superpoint))
    superpoint_center = np.zeros((spnum, 3), dtype='float32')
    for spID in np.unique(superpoint):
        spMask = np.where(superpoint == spID)[0]
        superpoint_center[spID] = xyz[spMask].mean(0)
    return superpoint_center


def normal_estimation(points, knn=33):
    o3d_pcd = to_o3d_pcd(points)
    o3d_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn))
    normals = np.asarray(o3d_pcd.normals)

    return normals


class Voxelize(object):
    def __init__(self,
                 voxel_size=0.05,
                 hash_type="fnv",
                 mode='train',
                 keys=("coord", "normal", "color", "label"),
                 return_discrete_coord=False,
                 return_min_coord=False):
        self.voxel_size = voxel_size
        self.hash = self.fnv_hash_vec if hash_type == "fnv" else self.ravel_hash_vec
        assert mode in ["train", "test"]
        self.mode = mode
        self.keys = keys
        self.return_discrete_coord = return_discrete_coord
        self.return_min_coord = return_min_coord

    def __call__(self, data_dict):
        assert "coord" in data_dict.keys()
        discrete_coord = np.floor(data_dict["coord"] / np.array(self.voxel_size)).astype(int)
        min_coord = discrete_coord.min(0) * np.array(self.voxel_size)
        discrete_coord -= discrete_coord.min(0)
        key = self.hash(discrete_coord)
        idx_sort = np.argsort(key)
        key_sort = key[idx_sort]
        _, inverse, count = np.unique(key_sort, return_inverse=True, return_counts=True)
        if self.mode == 'train':  # train mode
            # idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + np.random.randint(0, count.max(), count.size) % count
            idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1])
            idx_unique = idx_sort[idx_select]
            if self.return_discrete_coord:
                data_dict["discrete_coord"] = discrete_coord[idx_unique]
            if self.return_min_coord:
                data_dict["min_coord"] = min_coord.reshape([1, 3])
            for key in self.keys:
                data_dict[key] = data_dict[key][idx_unique]
            return data_dict

        elif self.mode == 'test':  # test mode
            data_part_list = []
            for i in range(count.max()):
                idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + i % count
                idx_part = idx_sort[idx_select]
                data_part = dict(index=idx_part)
                for key in data_dict.keys():
                    if key in self.keys:
                        data_part[key] = data_dict[key][idx_part]
                    else:
                        data_part[key] = data_dict[key]
                if self.return_discrete_coord:
                    data_part["discrete_coord"] = discrete_coord[idx_part]
                if self.return_min_coord:
                    data_part["min_coord"] = min_coord.reshape([1, 3])
                data_part_list.append(data_part)
            return data_part_list
        else:
            raise NotImplementedError

    @staticmethod
    def ravel_hash_vec(arr):
        """
        Ravel the coordinates after subtracting the min coordinates.
        """
        assert arr.ndim == 2
        arr = arr.copy()
        arr -= arr.min(0)
        arr = arr.astype(np.uint64, copy=False)
        arr_max = arr.max(0).astype(np.uint64) + 1

        keys = np.zeros(arr.shape[0], dtype=np.uint64)
        # Fortran style indexing
        for j in range(arr.shape[1] - 1):
            keys += arr[:, j]
            keys *= arr_max[j + 1]
        keys += arr[:, -1]
        return keys

    @staticmethod
    def fnv_hash_vec(arr):
        """
        FNV64-1A
        """
        assert arr.ndim == 2
        # Floor first for negative coordinates
        arr = arr.copy()
        arr = arr.astype(np.uint64, copy=False)
        hashed_arr = np.uint64(14695981039346656037) * np.ones(arr.shape[0], dtype=np.uint64)
        for j in range(arr.shape[1]):
            hashed_arr *= np.uint64(1099511628211)
            hashed_arr = np.bitwise_xor(hashed_arr, arr[:, j])
        return hashed_arr
