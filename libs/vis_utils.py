import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


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