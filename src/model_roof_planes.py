import numpy as np
from matplotlib import path as mpl_polygon
from open3d import geometry, utility
from pathlib import Path

from file_utils import read_image, read_metadata, read_ply
from visualize import visualize_point_cloud


def get_face_points(face: list[int], vertices: np.ndarray) -> np.ndarray:
    """
    Helper function to get face/polygon points as a 2D array of xy coordinates
    """
    face_points = vertices[face, :]
    return face_points


def image_to_world(vertices_pixels: np.ndarray, ppm: float, image_shape: tuple[int, int]) -> np.ndarray:
    """
    Helper function to convert xy image coordinates in pixels to xy world coordinates in the point cloud
    coordinate system. The point cloud is aligned with the image. The world origin is the center of the point cloud
    and +y is up in the world coordinates.

    ppm is short for pixels-per-meter
    """
    # center of image in subpixels
    rows, cols = image_shape
    c_row, c_col = (rows - 1) / 2, (cols - 1) / 2

    # offset by center of image, flip in the y-axis, and then convert from pixels to meters
    vertices_meters = (vertices_pixels - np.array([c_col, c_row])) * np.array([1.0, -1.0]) / ppm
    return vertices_meters


def lasso_points(face_polygon: np.ndarray, point_cloud: np.ndarray) -> np.ndarray:
    """
    Lasso 3D points with a 2D polygon
    """
    polygon_path = mpl_polygon.Path(vertices=face_polygon)
    points_mask = polygon_path.contains_points(point_cloud[:, :2], radius=0.0)
    interior_points = point_cloud[points_mask, :]
    return interior_points


def detect_plane_ransac(points: np.ndarray) -> tuple[float, float, float, float]:
    """
    Detect 3D plane equation given a set of 3D points. Use RANSAC to separate outliers.

    Returns tuple (a, b, c, d) where ax + by + cz + d = 0 is the 3D equation of a plane.
    """
    face_points_o3d = geometry.PointCloud()
    face_points_o3d.points = utility.Vector3dVector(points)

    # get plane with RANSAC using open3d utility
    plane, inliers = face_points_o3d.segment_plane(distance_threshold=0.2, ransac_n=3, num_iterations=500)
    return tuple(plane.tolist())


def model_roof_planes(
        point_cloud: np.ndarray,
        vertices: np.ndarray,
        faces: list[list[int]],
) -> list[tuple[float, float, float, float]]:

    for face in faces:
        face_polygon = vertices[face, :]
        face_points = lasso_points(face_polygon, point_cloud)

        plane_ransac = detect_plane_ransac(face_points[:, :3])

        # DEBUG: visualize point cloud points within a single face polygon
        visualize_point_cloud(face_points, polygon_2d=face_polygon, plane=plane_ransac)


if __name__ == "__main__":
    # data inputs
    data_path_ = Path('/Users/merrillmck/source/github/roof_modeling/data')
    uid_ = "ftlaud_1"

    # read data from files
    img_ = read_image(data_path_, uid_)
    point_cloud_ = read_ply(data_path_, uid_)
    vertices_pixels_, _, faces_, ppm_ = read_metadata(data_path_, uid_)
    vertices_ = image_to_world(vertices_pixels_, ppm_, img_.shape[:2])

    # model
    roof_planes_ = model_roof_planes(point_cloud_, vertices_, faces_)
