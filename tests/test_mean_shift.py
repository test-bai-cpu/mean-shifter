import numpy as np

from mean_shift.mean_shift import MeanShift


DATASET = np.array([[1, 1], [2, 1], [1, 0], [4, 7], [3, 5], [3, 6]])
TEST_DATASET = np.array([[0, 0], [2, 2]])


def test_flat_kernel():
    mean_shift_cluster = MeanShift(kernel="flat", kernel_parameters=[2], data_type="linear")
    mean_shift_cluster.fit(DATASET)
    result = [mean_shift_cluster.cluster_centers_.tolist(), mean_shift_cluster.labels_.tolist()]
    expected = [[[3.33, 6.], [1.33, 0.67]], [1, 1, 1, 0, 0, 0]]

    assert result == expected


def test_gaussian_kernel():
    mean_shift_cluster = MeanShift(kernel="gaussian", kernel_parameters=[0.25], data_type="linear")
    mean_shift_cluster.fit(DATASET)
    result = [mean_shift_cluster.cluster_centers_.tolist(), mean_shift_cluster.labels_.tolist()]
    expected = [[[3.27, 5.95], [1.33, 0.69]], [1, 1, 1, 0, 0, 0]]

    assert result == expected


def test_truncated_gaussian_kernel():
    mean_shift_cluster = MeanShift(
        kernel="truncated_gaussian", kernel_parameters=[1, 0.25], data_type="linear"
    )
    mean_shift_cluster.fit(DATASET)
    result = [mean_shift_cluster.cluster_centers_.tolist(), mean_shift_cluster.labels_.tolist()]
    expected = [[[1.32, 0.68], [3., 5.5], [4., 7.]], [0, 0, 0, 2, 1, 1]]

    assert result == expected


def test_mean_shift_cluster_predict():
    mean_shift_cluster = MeanShift(kernel="flat", kernel_parameters=[2], data_type="linear")
    mean_shift_cluster.fit(DATASET)
    result = mean_shift_cluster.predict(TEST_DATASET).tolist()
    expected = [1, 1]

    assert result == expected
