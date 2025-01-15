import numpy as np
from open3d import geometry, utility
from pathlib import Path
from typing import Literal

from file_utils import read_image, read_metadata, read_ply
from planar_regression import standardize_plane_np, planar_regression_lstsq
from point_cloud_utils import lasso_points, image_to_world
from visualize import visualize_roof_model, visualize_point_cloud


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
    plane = standardize_plane_np(plane)

    return tuple(plane.tolist())


def model_roof_planes(
        point_cloud: np.ndarray,
        vertices: np.ndarray,
        faces: list[list[int]],
        algorithm: Literal["ransac", "least_squares"] = "ransac",
) -> list[tuple[float, float, float, float]]:

    roof_planes = []
    for face in faces:
        # get points within each 2D roof polygon
        face_polygon = vertices[face, :]
        face_points = lasso_points(face_polygon, point_cloud)

        if algorithm == "ransac":
            plane = detect_plane_ransac(face_points[:, :3])
        else:
            plane = planar_regression_lstsq(face_points[:, :3])
        roof_planes.append(plane)

        # DEBUG: visualize point cloud points within a single face polygon
        visualize_point_cloud(face_points, polygon_2d=face_polygon, plane=plane)
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

    # model roof planes with RANSAC
    roof_planes_ransac_ = model_roof_planes(point_cloud_, vertices_, faces_, algorithm="ransac")

    # model roof planes with least squares fit
    # roof_planes_lstsq_ = model_roof_planes(point_cloud_, vertices_, faces_, algorithm="least_squares")

    # visualize roof
    polygons_2d_ = []
    for face_ in faces_:
        polygons_2d_.append(vertices_[face_, :])
    visualize_roof_model(point_cloud_, polygons_2d_, roof_planes_ransac_)
