import allantoolkit
import numpy as np


def test_return_type(data_with_gaps):
    """Test that function returns a numpy.array"""

    output = allantoolkit.utils.trim_data(data_with_gaps)

    assert isinstance(output, np.ndarray)


def test_full_array(data):
    """Should be the same """

    output = allantoolkit.utils.trim_data(data)

    assert np.array_equal(output, data)


def test_all_gaps(data_with_only_gaps):
    """"Test we get empty array if data is full of gaps"""

    output = allantoolkit.utils.trim_data(data_with_only_gaps)

    assert np.array_equal(output, np.array([]))


def test_array_with_holes():
    """ Only trim the array, do not remove NaN inside it """

    # Create artificial leading and trailing gaps
    array_with_holes = np.array([np.NaN, np.NaN, 1, 2, 3.4, 2.3, np.NaN, 9,
                                 12, 45, np.NaN])

    output = allantoolkit.utils.trim_data(array_with_holes)
    expected_trimmed_array = np.array([1, 2, 3.4, 2.3, np.NaN, 9, 12, 45])

    np.testing.assert_array_equal(expected_trimmed_array, output)

