from pathlib import Path
from src.file_utils import read_image
from visualize import visualize_image

VISUALIZE = True


def test_read_image():

    data_path = Path('/Users/merrillmck/source/github/roof_modeling/data')
    uid = "ftlaud_1"
    img = read_image(data_path, uid)

    assert len(img.shape) == 3  # RGB
    assert img.shape[2] == 3  # HxWx3

    if VISUALIZE:
        visualize_image(img)
