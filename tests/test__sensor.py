"""
Unit tests for functions in sensor.py
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import numpy as np

from hswfs.sensor import HSWFS
from hswfs.shifts import (
    generate_test_shifts,
    cut_to_circle,
    cut_out_circle
)


# -----------------------------------------------------------------------------
# TEST CASES
# -----------------------------------------------------------------------------


def test__x_shift() -> None:
    shifts = generate_test_shifts(test_case="x_shift", grid_size=8)
    sensor = HSWFS(relative_shifts=shifts)

    coefficients = sensor.fit_wavefront(n_zernike=9)
    expected = np.array([True if _ != 2 else False for _ in range(10)])

    assert np.array_equal(np.isclose(coefficients, 0), expected)


def test__y_shift() -> None:
    shifts = generate_test_shifts(test_case="y_shift", grid_size=8)
    sensor = HSWFS(relative_shifts=shifts)

    coefficients = sensor.fit_wavefront(n_zernike=9)
    expected = np.array([True if _ != 1 else False for _ in range(10)])

    assert np.array_equal(np.isclose(coefficients, 0), expected)


def test__defocus() -> None:
    shifts = generate_test_shifts(test_case="defocus", grid_size=8)
    sensor = HSWFS(relative_shifts=shifts)

    coefficients = sensor.fit_wavefront(n_zernike=9)
    expected = np.array([True if _ != 4 else False for _ in range(10)])

    assert np.array_equal(np.isclose(coefficients, 0), expected)


def test__defocus_circle() -> None:
    shifts = generate_test_shifts(test_case="defocus", grid_size=8)
    shifts = cut_to_circle(shifts)

    sensor = HSWFS(relative_shifts=shifts)

    coefficients = sensor.fit_wavefront(n_zernike=9)
    expected = np.array([True if _ != 4 else False for _ in range(10)])

    assert np.array_equal(np.isclose(coefficients, 0), expected)


def test__defocus_hole() -> None:
    shifts = generate_test_shifts(test_case="defocus", grid_size=8)
    shifts = cut_out_circle(shifts, 0.15)

    sensor = HSWFS(relative_shifts=shifts)

    coefficients = sensor.fit_wavefront(n_zernike=9)
    expected = np.array([True if _ != 4 else False for _ in range(10)])

    assert np.array_equal(np.isclose(coefficients, 0), expected)


def test__defocus_circle_hole() -> None:
    shifts = generate_test_shifts(test_case="defocus", grid_size=8)
    shifts = cut_to_circle(shifts)
    shifts = cut_out_circle(shifts, 0.15)

    sensor = HSWFS(relative_shifts=shifts)

    coefficients = sensor.fit_wavefront(n_zernike=9)
    expected = np.array([True if _ != 4 else False for _ in range(10)])

    assert np.array_equal(np.isclose(coefficients, 0), expected)


def test__x_shift_circle() -> None:
    shifts = generate_test_shifts(test_case="x_shift", grid_size=8)
    shifts = cut_to_circle(shifts)

    sensor = HSWFS(relative_shifts=shifts)

    coefficients = sensor.fit_wavefront(n_zernike=9)
    expected = np.array([True if _ != 2 else False for _ in range(10)])

    assert np.array_equal(np.isclose(coefficients, 0), expected)


def test__x_shift_hole() -> None:
    shifts = generate_test_shifts(test_case="x_shift", grid_size=8)
    shifts = cut_out_circle(shifts, 0.15)

    sensor = HSWFS(relative_shifts=shifts)

    coefficients = sensor.fit_wavefront(n_zernike=9)
    expected = np.array([True if _ != 2 else False for _ in range(10)])

    assert np.array_equal(np.isclose(coefficients, 0), expected)


def test__x_shift_circle_hole() -> None:
    shifts = generate_test_shifts(test_case="x_shift", grid_size=8)
    shifts = cut_to_circle(shifts)
    shifts = cut_out_circle(shifts, 0.15)

    sensor = HSWFS(relative_shifts=shifts)

    coefficients = sensor.fit_wavefront(n_zernike=9)
    expected = np.array([True if _ != 2 else False for _ in range(10)])

    assert np.array_equal(np.isclose(coefficients, 0), expected)


def test__y_shift_circle() -> None:
    shifts = generate_test_shifts(test_case="y_shift", grid_size=8)
    shifts = cut_to_circle(shifts)

    sensor = HSWFS(relative_shifts=shifts)

    coefficients = sensor.fit_wavefront(n_zernike=9)
    expected = np.array([True if _ != 1 else False for _ in range(10)])

    assert np.array_equal(np.isclose(coefficients, 0), expected)


def test__y_shift_hole() -> None:
    shifts = generate_test_shifts(test_case="y_shift", grid_size=8)
    shifts = cut_out_circle(shifts, 0.15)

    sensor = HSWFS(relative_shifts=shifts)

    coefficients = sensor.fit_wavefront(n_zernike=9)
    expected = np.array([True if _ != 1 else False for _ in range(10)])

    assert np.array_equal(np.isclose(coefficients, 0), expected)


def test__y_shift_circle_hole() -> None:
    shifts = generate_test_shifts(test_case="y_shift", grid_size=8)
    shifts = cut_to_circle(shifts)
    shifts = cut_out_circle(shifts, 0.15)

    sensor = HSWFS(relative_shifts=shifts)

    coefficients = sensor.fit_wavefront(n_zernike=9)
    expected = np.array([True if _ != 1 else False for _ in range(10)])

    assert np.array_equal(np.isclose(coefficients, 0), expected)
