import numpy as np
from pathlib import Path
from PIL import Image


# a "data" folder is organized with the following structure
#
# data/<uid>/ortho.png
# data/<uid>/dsm.ply
# data/<uid>/...
#


def read_image(data_path: Path, uid: str) -> np.ndarray:

    path = Path(data_path) / uid / "ortho.png"

    try:
        p_img = Image.open(path)
    except FileNotFoundError as e:
        print(e)
        return

    img = np.array(p_img)

    # cleanup
    p_img.close()

    return img
