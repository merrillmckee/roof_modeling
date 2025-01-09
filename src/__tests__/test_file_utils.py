from pathlib import Path
from src.file_utils import read_image, read_ply
from visualize import visualize_image, visualize_point_cloud

VISUALIZE = True


def test_read_image():

    data_path = Path('/Users/merrillmck/source/github/roof_modeling/data')
    uid = "ftlaud_1"
    img = read_image(data_path, uid)

    assert len(img.shape) == 3  # RGB
    assert img.shape[2] == 3  # HxWx3

    if VISUALIZE:
        visualize_image(img)


def test_read_ply():

    data_path = Path('/Users/merrillmck/source/github/roof_modeling/data')
    uid = "ftlaud_1"
    point_cloud = read_ply(data_path, uid)

    assert len(point_cloud.shape) == 2  # N points x C columns/features
    assert point_cloud.shape[1] == 9  # 9 features

    if VISUALIZE:
        visualize_point_cloud(point_cloud)
