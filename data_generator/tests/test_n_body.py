import pytest
import numpy as np
import pandas as pd

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


def test_nbody():
    nb = n_body.NBody(2, [3, 5, 7, 11])
    assert nb.m0 == 2
    assert nb.m == [3, 5, 7, 11]
    assert nb.n == 4


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

    nb.m.append(11)
    r = np.array([[1.5, 1, 0.3], [3, -1, -1.5], [-1.6, 0.2, 1], [1, 0, 0]])
    actual = nb.total_force(r)
    expected += np.array([-22, 0, 0])
    np.testing.assert_array_almost_equal(expected, actual)


def test_random_conformation():
    actual = n_body.random_conformations(4, 2, np.random.default_rng(42), 3)
    expected = np.array(
        [
            [
                [1.64373629133578, -0.3667293614876861, 2.1515875194682947],
                [1.1842081743561834, -2.434935912674103, 2.8537341098205355],
            ],
            [
                [1.5668382119421178, 1.7163858316617229, -2.2313182039467248],
                [-0.2976843726265972, -0.7752118546045126, 2.560589933091611],
            ],
            [
                [0.8631907204839873, 1.9365696796249798, -0.3395148070360132],
                [-1.6365676692913387, 0.3275087220950088, -2.6170964633749483],
            ],
            [
                [1.9657870319554924, 0.7899863947323891, 1.548526440512243],
                [-0.8728441912207898, 2.8241881463694196, 2.358726727933186],
            ],
        ]
    )
    np.testing.assert_array_almost_equal(expected, actual)


def test_coords2dataframe():
    r = [
        [[1, 2, 3], [4, 5, 6]],
        [[7, 8, 9], [10, 11, 12]],
        [[13, 14, 15], [16, 17, 18]],
        [[19, 20, 21], [22, 23, 24]],
    ]
    r = np.array(r)
    expected = {
        "x1": [1, 7, 13, 19],
        "y1": [2, 8, 14, 20],
        "z1": [3, 9, 15, 21],
        "x2": [4, 10, 16, 22],
        "y2": [5, 11, 17, 23],
        "z2": [6, 12, 18, 24],
    }
    expected = pd.DataFrame(expected)
    actual = n_body.coords2dataframe(r)
    pd.testing.assert_frame_equal(expected, actual)


def test_generate_random_observations():
    nb = n_body.NBody(2, [3, 5])
    actual = nb.generate_random_observations(
        4, np.random.default_rng(42), scale=2.5, epsilon=0.1
    )
    expected = {
        "x1": {
            0: 1.425456942194569,
            1: 1.3546509411333283,
            2: 0.645309901470417,
            3: 1.6782088803633604,
        },
        "y1": {
            0: -0.3666800596693449,
            1: 1.5238234728716111,
            2: 1.6089490515993365,
            3: 0.6207953238867324,
        },
        "z1": {
            0: 1.786333800302319,
            1: -1.8942667649946403,
            2: -0.33754713605316755,
            3: 1.3568906607059095,
        },
        "x2": {
            0: 0.8956008984542653,
            1: -0.27397836931519054,
            2: -1.3298435921396135,
            3: -0.6664172878512978,
        },
        "y2": {
            0: -2.098255362148243,
            1: -0.6520987165819322,
            2: 0.26035431885364013,
            3: 2.3309857977805515,
        },
        "z2": {
            0: 2.4147215488322704,
            1: 2.071719216059866,
            2: -2.1143780802675556,
            3: 1.9232712273970372,
        },
        "forces": {
            0: 2.00113252405678,
            1: 1.3330457063078516,
            2: 2.416168168305407,
            3: 1.9102618773856046,
        },
    }
    expected = pd.DataFrame(expected)
    pd.testing.assert_frame_equal(expected, actual)
