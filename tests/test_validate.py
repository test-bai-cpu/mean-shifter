import numpy as np

from mean_shift.validate import validate_data


DATASET = np.array([[0, 1], [np.pi, 1], [-1.2, 4]])


def test_validate_data():
    result = np.round(validate_data(DATASET, data_type="linear"), 2).tolist()
    expected = [[0., 1.], [3.14, 1.], [-1.2, 4.]]

    assert result == expected
