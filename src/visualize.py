import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use('MacOSX')


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
