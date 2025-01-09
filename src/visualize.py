import matplotlib.pyplot as plt
import numpy as np


def visualize_image(img: np.ndarray):
    """
    Visualize an image with matplotlib
    """
    fig, ax = plt.subplots(1, 1)
    ax.imshow(img)
    ax.axis('off')
    fig.tight_layout()
    plt.show()
