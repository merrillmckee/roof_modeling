import numpy as np
from matplotlib import path as mpl_polygon


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