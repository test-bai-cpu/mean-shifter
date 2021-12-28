import numpy as np

from mean_shift import utils


DATASET = np.array([[1, 1], [2, 1], [1, 0], [4, 7], [3, 5], [3, 6]])


def test_get_euclidean_distance():
    result = np.round(utils.get_euclidean_distance(np.array([1,1]), np.array([2,2])), 2)
    expected = 1.41

    assert result == expected


def test_get_flat_kernel():
    result = utils.get_flat_kernel(np.array([1, 1]), DATASET, [2]).tolist()
    expected = [1., 1., 1., 0., 0., 0.]

    assert result == expected


def test_get_gaussian_kernel():
    result = np.round(utils.get_gaussian_kernel(np.array([1, 1]), DATASET, [0.25]), 2).tolist()
    expected = [1., 0.78, 0.78, 0., 0.01, 0.]

    assert result == expected


def test_get_truncated_gaussian_kernel():
    result = np.round(utils.get_truncated_gaussian_kernel(np.array([3, 6]), DATASET, [2, 0.25]), 2).tolist()
    expected = [0., 0., 0., 0.61, 0.78, 1.]

    assert result == expected
