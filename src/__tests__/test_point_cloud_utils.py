import numpy as np
import numpy.testing as npt

from point_cloud_utils import image_to_world, lasso_points


def test_image_to_world():
    image_shape = (201, 301)  # 201 rows, 301 columns
    vertices_pixels = np.array([
        [150.0, 100.0],  # xy center of image -> (0, 0) origin of point cloud
        [160.0, 110.0],  # +1 meter right and down -> (1, -1)
        [160.0, 090.0],  # +1 meter right and up -> (1, +1)
    ])
    ppm = 10.0  # 10 pixels / meter

    vertices_world = image_to_world(vertices_pixels, ppm, image_shape)

    assert vertices_world.shape == (3, 2)
    npt.assert_array_equal(vertices_world[0, :], [0, 0])   # verify center of image
    npt.assert_array_equal(vertices_world[1, :], [1, -1])  # verify +1 meter right and down from center
    npt.assert_array_equal(vertices_world[2, :], [1, 1])   # verify +1 meter right and up from center


def test_lasso_points():
    face_polygon = np.array([
        [0.0, 0.0],  # unit square
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
    ])
    point_cloud = np.array([
        [0.1, 0.1, 11.0],  # interior
        [0.9, 0.9, 12.0],  # interior
        [1.1, 0.1, 13.0],  # exterior
        [0.1, 1.1, 14.0],  # exterior
        [-0.1, 0.1, 15.0],  # exterior
        [0.1, -0.1, 16.0],  # exterior
    ])
    interior_points = lasso_points(face_polygon, point_cloud)
    npt.assert_array_equal(interior_points, point_cloud[:2, :])  # first two points are lassod
