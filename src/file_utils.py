import numpy as np
from pathlib import Path
from PIL import Image
from plyfile import PlyData


# a "data" folder is organized with the following structure
#
# data/<uid>/ortho.png
# data/<uid>/dsm.ply
# data/<uid>/...
#


def read_image(data_path: Path, uid: str) -> np.ndarray:

    img_path = Path(data_path) / uid / "ortho.png"

    try:
        p_img = Image.open(img_path)
    except FileNotFoundError as e:
        print(e)
        return

    img = np.array(p_img)

    # cleanup
    p_img.close()

    return img


def read_ply(data_path: Path, uid: str) -> np.ndarray:

    ply_path = Path(data_path) / uid / "dsm.ply"

    try:
        plydata = PlyData.read(ply_path)
        vertices = plydata['vertex']
        x = np.array(vertices[:]['x'])
        y = np.array(vertices[:]['y'])
        z = np.array(vertices[:]['z'])
        nx = np.array(vertices[:]['nx'])
        ny = np.array(vertices[:]['ny'])
        nz = np.array(vertices[:]['nz'])
        r = np.array(vertices[:]['red'])
        g = np.array(vertices[:]['green'])
        b = np.array(vertices[:]['blue'])
        point_cloud = np.column_stack((x, y, z, nx, ny, nz, r, g, b))
    except FileNotFoundError as e:
        print(e)
        return

    return point_cloud
