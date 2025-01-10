import numpy as np
from open3d import geometry, utility
from pathlib import Path

from file_utils import read_image, read_metadata, read_ply
from point_cloud_utils import lasso_points, image_to_world
from visualize import visualize_roof_model


def detect_plane_ransac(points: np.ndarray) -> tuple[float, float, float, float]:
    """
    Detect 3D plane equation given a set of 3D points. Use RANSAC to separate outliers.

    Returns tuple (a, b, c, d) where ax + by + cz + d = 0 is the 3D equation of a plane.
    """
    face_points_o3d = geometry.PointCloud()
    face_points_o3d.points = utility.Vector3dVector(points)

    # get plane with RANSAC using open3d utility
    plane, inliers = face_points_o3d.segment_plane(distance_threshold=0.2, ransac_n=3, num_iterations=500)

    # ensure the plane coefficients have a positive 'c'; this means normals point up
    c = plane[2]
    if c != 0:
        plane /= np.sign(c)

    return tuple(plane.tolist())


def model_roof_planes(
        point_cloud: np.ndarray,
        vertices: np.ndarray,
        faces: list[list[int]],
) -> list[tuple[float, float, float, float]]:

    roof_planes = []
    for face in faces:
        # get points within each 2D roof polygon
        face_polygon = vertices[face, :]
        face_points = lasso_points(face_polygon, point_cloud)

        plane_ransac = detect_plane_ransac(face_points[:, :3])
        roof_planes.append(plane_ransac)

        # DEBUG: visualize point cloud points within a single face polygon
        # visualize_point_cloud(face_points, polygon_2d=face_polygon, plane=plane_ransac)
    return roof_planes


if __name__ == "__main__":
    # data inputs
    data_path_ = Path('/Users/merrillmck/source/github/roof_modeling/data')
    uid_ = "ftlaud_1"

    # read data from files
    img_ = read_image(data_path_, uid_)
    point_cloud_ = read_ply(data_path_, uid_)
    vertices_pixels_, _, faces_, ppm_ = read_metadata(data_path_, uid_)
    vertices_ = image_to_world(vertices_pixels_, ppm_, img_.shape[:2])

    # model roof planes
    roof_planes_ = model_roof_planes(point_cloud_, vertices_, faces_)

    # visualize roof
    polygons_2d_ = []
    for face_ in faces_:
        polygons_2d_.append(vertices_[face_, :])
    visualize_roof_model(point_cloud_, polygons_2d_, roof_planes_)
