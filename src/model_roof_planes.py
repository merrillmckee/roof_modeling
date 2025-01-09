import numpy as np
from matplotlib import path as Polygon
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


def model_roof_planes(
        point_cloud: np.ndarray,
        vertices: np.ndarray,
        faces: list[list[int]],
) -> list[tuple[float, float, float, float]]:

    for face in faces:
        face_polygon = vertices[face, :]
        polygon_path = Polygon.Path(vertices=face_polygon)
        points_mask = polygon_path.contains_points(point_cloud[:, :2], radius=0.0)
        face_points = point_cloud[points_mask, :]

        # DEBUG: visualize point cloud points within a single face polygon
        visualize_point_cloud(face_points)


if __name__ == "__main__":
    # data
    data_path = Path('/Users/merrillmck/source/github/roof_modeling/data')
    uid = "ftlaud_1"

    # read data from file
    img = read_image(data_path, uid)
    point_cloud = read_ply(data_path, uid)
    vertices_pixels, _, faces, ppm = read_metadata(data_path, uid)
    vertices = image_to_world(vertices_pixels, ppm, img.shape[:2])

    # model
    roof_planes = model_roof_planes(point_cloud, vertices, faces)
