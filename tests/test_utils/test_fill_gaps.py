import allantoolkit
import numpy as np


def test_return_type(data_with_gaps):
    """Test that function returns a numpy.array."""

    output = allantoolkit.utils.fill_gaps(data_with_gaps)

    assert isinstance(output, np.ndarray)


def test_return_size_no_edge_gaps(data_with_gaps):
    """Test that output data has same dimensionality if there are no leading
    or trailing gaps"""

    # Make sure no leading or trailing NaNs
    data_with_gaps[0], data_with_gaps[-1] = 1, 1

    output = allantoolkit.utils.fill_gaps(data_with_gaps)

    assert output.size == data_with_gaps.size


def test_return_size_edge_gaps(data_with_gaps):
    """Test that output data has reduced dimensionality if there are leading
    or trailing gaps"""

    # Create artificial leading and trailing gaps
    data_with_gaps[0], data_with_gaps[1], data_with_gaps[2] = np.NaN, np.NaN, 1
    data_with_gaps[-1], data_with_gaps[-2] = np.NaN, 2

    output = allantoolkit.utils.fill_gaps(data_with_gaps)

    assert output.size == data_with_gaps.size - 3


def test_is_filled(data_with_gaps):
    """Test that no gaps are left at the end of the conversion"""

    output = allantoolkit.utils.fill_gaps(data_with_gaps)

    # optimised way to check if any NaNs in arrays:
    # https://stackoverflow.com/questions/6736590/fast-check-for-nan-in-numpy
    assert ~np.isnan(np.dot(output, output))


def test_invariance(data):
    """Test that input data is preserved if input data has no gaps"""

    output = allantoolkit.utils.fill_gaps(data)

    assert np.array_equal(output, data)


def test_all_gaps():
    """Test we get empty array if data is full of gaps"""

    data = np.empty(100)
    data[:] = np.NaN

    output = allantoolkit.utils.fill_gaps(data)

    assert np.array_equal(output, np.array([]))


