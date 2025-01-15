import math
import numpy as np
from typing import Union


##############################
# All plane equations are represented by tuple (a, b, c, d) using equation
#   ax + by + cz + d = 0
##############################


def in_plane(points: np.ndarray, plane: tuple[float, float, float, float]) -> bool:
    """
    Returns True if all given points are on the given plane
    """
    x, y, z = points[:, :3].T
    a, b, c, d = plane
    return np.all(np.isclose(a * x + b * y + c * z + d, 0))


def calculate_plane_equation_3_points(points: np.ndarray) -> Union[None, tuple[float, float, float, float]]:
    """
    Given 3 points, calculate the plane equation if the points are not collinear
    """
    x1, x2, x3 = points[:3, :3]
    v1 = x1 - x2
    v2 = x1 - x3
    normal = np.cross(v1, v2)
    if np.all(normal==0):
        return None
    d = -np.dot(normal, x1)
    plane = np.array(normal.tolist() + [d])
    plane = standardize_plane_np(plane)
    return plane


def standardize_plane(plane: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """
    Standardize plane equation so that c >= 0 and the normal vector <a, b, c> is a unit normal vector
    """
    # ensure the plane coefficients have a positive 'c'; this means normals "point up"
    a, b, c, d = plane
    if c < 0:
        plane = [-1 * x for x in plane]

    # normalize normal vector <a, b, c> to be a unit normal vector
    norm = math.sqrt(a * a + b * b + c * c)
    plane = [x / norm for x in plane]

    return plane


def standardize_plane_np(plane: np.ndarray) -> np.ndarray:
    """
    Standardize plane equation so that c >= 0 and the normal vector <a, b, c> is a unit normal vector
    """
    # ensure the plane coefficients have a positive 'c'; this means normals "point up"
    c = plane[2]
    if c < 0:
        plane *= -1.0

    # normalize normal vector <a, b, c> to be a unit normal vector
    plane /= np.linalg.norm(plane[:3])

    return plane


def planar_regression_lstsq(points: np.ndarray) -> tuple[float, float, float, float]:
    """
    Model the 3D plane that has least squared error through a set of points

    Returns (a, b, c, d) where ax + by + cz + d = 0, c is non-negative, and <a, b, c> is the unit normal vector

    Intermediate math
    ax + by + cz + d = 0
    ax + by + cz = -d  (rearrange)
    ax + by + cz = 1  (let d == -1)

    Ax=y
    A: Nx3 matrix where the 3 columns are x, y, z coordinates
    x: 3x1 matrix where the 3 unknowns are a, b, c
    y: Nx1 matrix of 1's

    General least squared error solution for Ax=y is x = inv(A' * A) * A' * y
      - where A' is the transpose of matrix A and inv(A) is the inverse of matrix A
      - this solution requires (A' * A) to be invertible
    """
    # setup
    n = len(points)
    A = points[:, :3]  # columns: x, y, z
    y = np.ones(shape=(n, 1), dtype=float)

    gram_matrix = A.T @ A
    rank = np.linalg.matrix_rank(gram_matrix)
    if rank < 2:
        raise ValueError(f"Unable to determine best fit plane; matrix rank is {rank}")
    if rank == 2:
        # some exact fit planes have a rank 2 singular matrix for AT @ A
        plane = calculate_plane_equation_3_points(points[:3, :3])
        if in_plane(points, plane):
            return standardize_plane_np(plane).tolist()
        else:
            raise ValueError(f"Unable to determine best fit plane; matrix rank is {rank}")

    # solve for unknowns y: <a, b, c>
    plane_normal = (np.linalg.inv(gram_matrix) @ A.T @ y).squeeze()

    # solution <a, b, c> -> (a, b, c, d)  where d == -1
    plane = np.array(plane_normal.tolist() + [-1.0])  # from letting d == -1
    plane = standardize_plane_np(plane)

    return plane.tolist()
