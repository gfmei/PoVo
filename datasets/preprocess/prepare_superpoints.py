from segmentator import segment_point
from torch_cluster import knn_graph


def gen_superpoints(points, normals, k=50, kThresh=0.01, segMinVerts=20):
    edges = knn_graph(points, k=k).T
    superpoint = segment_point(points, normals, edges, kThresh, segMinVerts)

    return superpoint
