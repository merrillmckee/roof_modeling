import pytest

import math
import numpy as np
import numpy.testing as npt

from planar_regression import in_plane, planar_regression_lstsq, calculate_plane_equation_3_points, standardize_plane, \
    standardize_plane_np


##############################
# All plane equations are represented by tuple (a, b, c, d) using equation
#   ax + by + cz + d = 0
##############################


def test_in_plane():
    points = np.array([
        [0.0, 0.0, 0.0],  # unit square defines the ground plane
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ])
    plane = (0.0, 0.0, 1.0, 0.0)  # ground plane
    result = in_plane(points, plane)
    assert result == True

    plane = (1.0, 1.0, 1.0, 1.0)  # incorrect plane
    result = in_plane(points, plane)
    assert result == False


def test_calculate_plane_equation_3_points():
    points = np.array([
        [0.0, 0.0, 0.0],  # unit triangle defines the ground plane
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
    ])
    expected_plane = np.array((0.0, 0.0, 1.0, 0.0))  # ground plane
    result_plane = calculate_plane_equation_3_points(points)
    npt.assert_almost_equal(result_plane, expected_plane)


def test_standardize_plane():
    plane = (-1, -1, -1, -1)
    z = 1 / math.sqrt(3)
    expected_plane = (z, z, z, z)
    actual_plane = standardize_plane(plane)
    for x, y in zip(actual_plane, expected_plane):
        assert x == pytest.approx(y)


def test_standardize_plane_np():
    # arrange
    plane = np.array((-1, -1, -1, -1), dtype=float)
    z = 1 / math.sqrt(3)
    expected_plane = np.array((z, z, z, z))

    # act
    actual_plane = standardize_plane_np(plane)

    # assert
    npt.assert_almost_equal(actual_plane, expected_plane)


def test_planar_regression_lstsq():
    # arrange
    points = np.array([
        [0.0, 0.0, 0.000003],  # 3D diamond pattern with approximate plane fit due to noise added to z-values
        [1.0, 0.0, 1.000004],
        [0.0, 1.0, 1.000005],
        [1.0, 1.0, 2.000006],
    ])
    z = 1 / math.sqrt(3)
    expected_plane = (-z, -z, z, 0)

    # act
    plane = planar_regression_lstsq(points)

    # assert
    assert in_plane(points,plane)
    npt.assert_almost_equal(plane, expected_plane, decimal=5)


def test_planar_regression_lstsq_exact():
    # arrange
    points = np.array([
        [0.0, 0.0, 0.0],  # 3D diamond pattern with exact plane fit
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [1.0, 1.0, 2.0],
    ])
    z = 1 / math.sqrt(3)
    expected_plane = (-z, -z, z, 0)

    # act
    plane = planar_regression_lstsq(points)

    # assert
    assert in_plane(points,plane)
    npt.assert_almost_equal(plane, expected_plane)


def test_planar_regression_lstsq_near_unit_xy_square():
    # this unit test should pass in most cases as (A' @ A) should be invertible
    # some noise added to the z-values
    points = np.array([
        [0.0, 0.0, 0.0000001],  # unit square defines the ground plane
        [1.0, 0.0, 0.0000002],
        [1.0, 1.0, -0.0000001],
        [0.0, 1.0, -0.00000003],
    ])
    plane = planar_regression_lstsq(points)
    npt.assert_almost_equal(plane, (0.0, 0.0, 1.0, 0.0), decimal=5)  # ground plane


def test_planar_regression_lstsq_exact_unit_xy_square():
    # this unit test will not pass if the algorithm relies on the inverse of singular (A.T @ A)
    points = np.array([
        [0.0, 0.0, 0.0],  # unit square defines the ground plane
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ])
    plane = planar_regression_lstsq(points)
    npt.assert_almost_equal(plane, (0.0, 0.0, 1.0, 0.0), decimal=7)  # ground plane
