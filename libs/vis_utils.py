import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import open3d as o3d


SCANNET_COLOR_MAP_20 = {-1: (0., 0., 0.), 0: (174., 199., 232.), 1: (152., 223., 138.), 2: (31., 119., 180.),
                        3: (255., 187., 120.), 4: (188., 189., 34.), 5: (140., 86., 75.),
                        6: (255., 152., 150.), 7: (214., 39., 40.), 8: (197., 176., 213.), 9: (148., 103., 189.),
                        10: (196., 156., 148.), 11: (23., 190., 207.), 12: (247., 182., 210.),
                        13: (219., 219., 141.), 14: (255., 127., 14.), 15: (158., 218., 229.), 16: (44., 160., 44.),
                        17: (112., 128., 144.), 18: (227., 119., 194.), 19: (82., 84., 163.)}


def visualize_feature_map(feature_map, w, h):
    # Reshape the feature map to [576, 1024]
    # reshaped_features = feature_map.reshape(-1, 1024)

    # Apply PCA to reduce dimensions to 3
    pca = PCA(n_components=3)
    reduced_features = pca.fit_transform(feature_map)

    # Rescale features to [0, 1] for RGB visualization
    min_max_scaled = (reduced_features - reduced_features.min(axis=0)) / (reduced_features.max(axis=0) - reduced_features.min(axis=0))

    # Reshape back to [24, 24, 3] for visualization
    rgb_image = min_max_scaled.reshape(w, h, 3)

    # Visualize the RGB image
    plt.imshow(rgb_image)
    plt.axis('off')  # Hide axis
    plt.show()


def visualize_image(image):
    # Display the image
    plt.imshow(image)
    plt.axis('off')  # Optional: to remove axis ticks and labels
    plt.show()


def feature_cluster_visualization(feature_map, n_clusters=6, w=18, h=24):

    # Assuming 'feature_map' is your input with shape [W*H, D]
    # Step 1: The feature_map is already in the required shape [W*H, D]

    # Step 2: Apply K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(feature_map)

    # Step 3: Visualization
    # Map cluster labels back to the original image size
    cluster_labels = kmeans.labels_
    clustered_image = cluster_labels.reshape(w, h)

    # Plot the clustered image
    plt.figure(figsize=(6, 6))
    plt.imshow(clustered_image, cmap='viridis')  # Using 'viridis' colormap to distinguish clusters
    plt.axis('off')
    plt.title('Clustered Image Visualization')
    plt.show()


def creat_labeled_point_cloud(points, labels, name):
    # Create a point cloud object from the points
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Map the integer labels to colors using a colormap
    # Normalize the labels to be in the range [0, 1] for the colormap
    # normalized_labels = labels / max(labels.max(), 1)  # Avoid division by zero
    # colors = plt.get_cmap("viridis")(normalized_labels)
    # Generate a colormap with 21 distinct colors
    cmap = plt.get_cmap('tab20c', len(np.unique(labels)))  # 'tab20b' has 20 distinct colors, adjust as needed for 21
    # Map hard labels to colors using the colormap
    colors = np.array([cmap(i)[:3] for i in labels])  # Extract RGB components
    # colors = plt.get_cmap("tab20")(normalized_labels)
    # Open3D expects colors in RGB format, in the range [0, 1]
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])  # Exclude alpha channel
    o3d.io.write_point_cloud(name + f'.ply', pcd)


def get_super_point_cloud(mesh_file, xyz):
    mesh = trimesh.load_mesh(mesh_file)
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


def get_colored_point_cloud_pca_sep(xyz, feature, name):
    """N x D"""
    pca = PCA(n_components=3)
    pca_gf = pca.fit_transform(feature)
    pca_gf = (pca_gf + np.abs(pca_gf.min(0))) / (pca_gf.ptp(0) + 1e-4)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz[:, 0:3])
    pcd.colors = o3d.utility.Vector3dVector(pca_gf)
    o3d.io.write_point_cloud(name + f'.ply', pcd)


def get_colored_point_cloud_from_labels(xyz, soft_labels, name):
    # Convert soft labels to hard labels
    hard_labels = np.argmax(soft_labels, axis=1)
    unique_labels = np.unique(hard_labels)
    # Generate a colormap with 21 distinct colors
    cmap = plt.get_cmap('tab20', len(unique_labels))  # 'tab20b' has 20 distinct colors, adjust as needed for 21

    # Map hard labels to colors using the colormap
    colors = np.array([cmap(i)[:3] for i in hard_labels])  # Extract RGB components

    # Create and color the point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Save the point cloud
    o3d.io.write_point_cloud(name + f'.ply', pcd)


def draw_point_cloud(points, colors, name):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(name + f'.ply', pcd)


def get_labeled_point_cloud_from_features(xyz, features, k=6, name='pca'):
    kmeans = KMeans(n_clusters=k, random_state=0).fit(features)
    labels = kmeans.labels_
    # Create an Open3D point cloud object from the points
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    # Assign colors to each point based on its cluster label for visualization
    colors = plt.get_cmap("tab20")(labels / k)  # Using matplotlib's tab10 colormap
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])  # Open3D expects RGB colors
    # Save the point cloud
    o3d.io.write_point_cloud(name + f'.ply', pcd)