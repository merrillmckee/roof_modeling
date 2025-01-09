import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

matplotlib.use('MacOSX')


COLORS = [
    [0, 255, 255],  # cyan
    [255, 0, 255],  # magenta
    [255, 0, 0],    # red
    [0, 255, 0],    # green
    [0, 0, 255],    # blue
    [160, 32, 255], # purple
    [96, 255, 128], # yellow-green
    [255, 160, 16], # orange
    [255, 208, 160], # pale pink
]


def to_tuple(arr: np.ndarray) -> tuple:
    """
    Helper to convert an array to tuple since Pillow likes tuple inputs
    """
    return tuple(arr.tolist())


def visualize_image(img: np.ndarray):
    """
    Visualize an image with matplotlib
    """
    fig, ax = plt.subplots(1, 1)
    ax.imshow(img)
    ax.axis('equal')
    ax.axis('off')
    fig.tight_layout()
    plt.show()

    # cleanup
    plt.close(fig)


def visualize_point_cloud(point_cloud: np.ndarray):
    """
    Visualize a 3D point cloud with matplotlib

    Note: Can be very slow
    """
    x = point_cloud[:, 0]
    y = point_cloud[:, 1]
    z = point_cloud[:, 2]
    colors = point_cloud[:, 6:9] / 255.0

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c=colors)
    ax.axis('equal')
    ax.axis('off')
    fig.tight_layout()
    plt.show()

    # cleanup
    plt.close(fig)


def visualize_2d_features_image_overlay(
        img: np.ndarray,
        vertices: np.ndarray,
        edges: np.ndarray,
        faces: list[list[int]],
):
    """
    Visualize 2D features overlaid onto an image with Pillow
    """
    radius = 10  # pixels
    p_img = Image.fromarray(img)
    pencil = ImageDraw.Draw(p_img, mode="RGBA")

    # vertices
    for v in vertices:
        xy_1 = to_tuple(v - radius)
        xy_2 = to_tuple(v + radius)
        pencil.ellipse(xy=[xy_1, xy_2], fill=None, outline="yellow", width=1)

    # edges
    for e in edges:
        xy_1 = to_tuple(vertices[e[0], :])
        xy_2 = to_tuple(vertices[e[1], :])
        pencil.line(xy=[xy_1, xy_2], fill="yellow", width=0)

    # faces
    alpha = 60
    F = len(COLORS)
    for i, face in enumerate(faces):
        xys = []
        for v in face:
            xys.append(to_tuple(vertices[v, :]))
        pencil.polygon(xy=xys, fill=tuple(COLORS[i%F] + [alpha]))  # very transparent yellow

    # display
    p_img.show()

    # cleanup
    p_img.close()
