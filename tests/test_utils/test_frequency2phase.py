import allantoolkit
import numpy as np

# Randomise fixed parameters
RATE = np.random.random()*100


def test_return_type(data):
    """Test that function returns a numpy.array"""

    output = allantoolkit.utils.frequency2phase(data, rate=RATE)

    assert isinstance(output, np.ndarray)


def test_return_size(data):
    """Test that output phase data has increased dimensionality due to
    integration."""

    # using data without gaps to make check easier
    output = allantoolkit.utils.frequency2phase(data, rate=RATE)

    assert output.size == data.size + 1


def test_is_filled(data_with_gaps):
    """Test that no gaps are left at the end of the conversion"""

    output = allantoolkit.utils.frequency2phase(data_with_gaps, rate=RATE)

    # optimised way to check if any NaNs in arrays:
    # https://stackoverflow.com/questions/6736590/fast-check-for-nan-in-numpy
    assert ~np.isnan(np.dot(output, output))


def test_leading_phase(data_with_gaps):
    """Test that arbitrary 0 phase has been inserted at beginning of dataset"""

    output = allantoolkit.utils.frequency2phase(data_with_gaps, rate=RATE)

    assert output[0] == 0


def test_all_gaps(data_with_only_gaps):
    """"Test we get empty array if data is full of gaps"""

    output = allantoolkit.utils.frequency2phase(data_with_only_gaps, rate=RATE)

    assert np.array_equal(output, np.array([]))


def test_almost_all_gaps(data_with_only_gaps):
    """Test behaviour if one and only value in dataset"""

    # fill one value
    data_with_only_gaps[5] = 1

    output = allantoolkit.utils.frequency2phase(data_with_only_gaps, rate=RATE)

    output_theory = np.cumsum(np.array(1.)) * 1 / RATE
    output_theory = np.insert(output_theory, 0, 0)

    assert np.array_equal(output, output_theory)
