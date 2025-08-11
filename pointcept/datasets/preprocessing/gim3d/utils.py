import copy
import open3d as o3d
import numpy as np


def remove_unconnected_parts(
    mesh: o3d.geometry.TriangleMesh, threshold=10000
) -> o3d.geometry.TriangleMesh:
    """
    Remove unconnected parts from 3d mesh and return the filtered mesh
    :param mesh: starting mesh
    :return: the filtered mesh
    """

    # Create clusters
    triangle_clusters, cluster_n_triangles, cluster_area = (
        mesh.cluster_connected_triangles()
    )
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)

    # Create filtered mesh
    filtered_mesh = copy.deepcopy(mesh)
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < threshold
    filtered_mesh.remove_triangles_by_mask(triangles_to_remove)

    # Remove duplicated vertices and triangles
    filtered_mesh.remove_duplicated_vertices()
    filtered_mesh.remove_duplicated_triangles()

    # Remove unreferenced vertices
    filtered_mesh.remove_unreferenced_vertices()

    return filtered_mesh


def create_point_cloud_from_vertices(
    pc_vertices: np.ndarray, pc_normals: np.ndarray = None, pc_colors: list = None
) -> o3d.geometry.PointCloud:
    """
    Given vertices and faces, this function creates a triangular mesh.
    :param pc_vertices: vertices of the point cloud
    :param pc_normals: normals of the point cloud (optional)
    :param pc_colors: colors of the point cloud (optional)
    :return: point cloud
    """

    # Create new point cloud
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(pc_vertices)

    # Apply normals if exists
    if pc_normals is not None:
        point_cloud.normals = o3d.utility.Vector3dVector(pc_normals)

    # Apply colors if exists, otherwise apply GRAY
    if pc_colors is not None:
        point_cloud.colors = o3d.utility.Vector3dVector(pc_colors)
    else:
        colors = [GRAY] * len(point_cloud.points)
        point_cloud.colors = o3d.utility.Vector3dVector(colors)

    return point_cloud
