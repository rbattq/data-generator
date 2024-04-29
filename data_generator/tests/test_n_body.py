import pytest
import numpy as np

from .. import n_body


def test_compute_force():
    m0 = 2
    m1 = 3
    r = np.array([1, 0, 0])

    expected = np.array([-6, 0, 0])
    actual = n_body.compute_force(m0, m1, r)
    np.testing.assert_array_equal(expected, actual)

    r = np.array([0, 1, 0])
    expected = np.array([0, -6, 0])
    actual = n_body.compute_force(m0, m1, r)
    np.testing.assert_array_equal(expected, actual)

    r = np.array([0, 0, -1])
    expected = np.array([0, 0, 6])
    actual = n_body.compute_force(m0, m1, r)
    np.testing.assert_array_equal(expected, actual)

    r = np.array([3, -4, 0])

    r2 = 25
    r_hat = r / 5
    expected = -6 / r2 * r_hat
    actual = n_body.compute_force(m0, m1, r)
    np.testing.assert_array_almost_equal(expected, actual)


def test_total_force():
    nb = n_body.NBody(2, [3, 5, 7])
    r = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
    actual = nb.total_force(r)
    expected = np.array([-6, 10, -14])
    np.testing.assert_array_equal(expected, actual)

    r = np.array([[1, 1, 0], [0, -1, -1], [-1, 0, 1]])
    actual = nb.total_force(r)
    expected = np.array([2.82842712474619, 1.414213562373095, -1.414213562373095])
    np.testing.assert_array_almost_equal(expected, actual)

    r = np.array([[1.5, 1, 0.3], [3, -1, -1.5], [-1.6, 0.2, 1]])
    actual = nb.total_force(r)
    expected = np.array([1.1052651820707768, -1.1596390090089517, -1.9946552655847771])
    np.testing.assert_array_almost_equal(expected, actual)
