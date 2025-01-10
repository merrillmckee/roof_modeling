import numpy as np

from model_roof_planes import detect_plane_ransac


def test_detect_plane_ransac():
    points = np.array([
        [0.0, 0.0, 0.0],  # unit square defines the ground plane
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.1, 0.2, 10.3],  # outlier 1
        [0.4, 0.5, -10.6],  # outlier 2
    ])
    plane = detect_plane_ransac(points)
    assert plane == (0.0, 0.0, 1.0, 0.0)  # ground plane or z=0
