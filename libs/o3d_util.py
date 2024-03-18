import numpy as np
import open3d as o3d
import torch
import trimesh

try:
    import segmentator
except Exception as e:
    pass


def to_o3d_pcd(pcd, normal=None, est_normal=False, radius=0.05):
    '''
    Transfer a point cloud of numpy.ndarray to open3d point cloud
    :param radius:
    :param normal:
    :param est_normal:
    :param pcd: point cloud of numpy.ndarray in shape[N, 3]
    :return: open3d.geometry.PointCloud()
    '''
    if torch.is_tensor(pcd):
        pcd = pcd.cpu().numpy()
    if torch.is_tensor(normal):
        normal = normal.cpu().numpy()
    pcd_ = o3d.geometry.PointCloud()
    pcd_.points = o3d.utility.Vector3dVector(pcd)
    if est_normal:
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=33))
    if normal is not None:
        pcd_.normals = o3d.utility.Vector3dVector(normal)
    return pcd_


def generate_mesh_from_pcd(xyz, normal=None):
    pcd = to_o3d_pcd(xyz, normal)
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


def get_super_point_cloud(mesh):
    # mesh = trimesh.load_mesh(mesh_file)
    _vertices = torch.from_numpy(mesh.vertices.astype(np.float32))
    _faces = torch.from_numpy(mesh.faces.astype(np.int64))
    superpoint = segmentator.segment_mesh(_vertices, _faces).numpy()
    # creat_labeled_point_cloud(_vertices, superpoint, 'label_spts')
    return _vertices, superpoint


def get_spt_centers(points, ids):
    spnum = len(np.unique(ids))
    superpoint_center = np.zeros((spnum, 3), dtype='float32')
    uni_ids = np.unique(ids)
    # for spID in uni_ids:
    #     spMask = np.where(ids == spID)[0]
    #     superpoint_center[spID] = points[spMask].mean(0)

    return superpoint_center, uni_ids


def normal_estimation(points, knn=33):
    o3d_pcd = to_o3d_pcd(points)
    o3d_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn))
    normals = np.asarray(o3d_pcd.normals)

    return normals
