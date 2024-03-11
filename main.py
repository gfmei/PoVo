import open3d as o3d

from libs.o3d_util import generate_mesh_from_pcd, get_super_point_cloud


def print_hi(knn=33):
    # Path to your PLY file
    ply_path = 'orig.ply'
    # Load the PLY file
    pcd = o3d.io.read_point_cloud(ply_path)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn))
    xyz = pcd.points
    print(pcd.normals)
    # To visualize the point cloud (optional)
    o3d.visualization.draw_geometries([pcd])
    mesh = generate_mesh_from_pcd(xyz, normal=None)
    # spts = get_super_point_cloud(mesh, xyz)
    # print(spts)
    mesh.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi(33)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
