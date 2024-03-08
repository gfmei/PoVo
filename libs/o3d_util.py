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


def generate_mesh_from_pcd(xyz, normal=None, radius=0.05):
    pcd = to_o3d_pcd(xyz)
    if normal is None:
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=33))
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
