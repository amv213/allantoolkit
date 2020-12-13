import allantoolkit
import numpy as np

# Randomise fixed parameters
RATE = np.random.random()*100


def test_return_type(data):
    """Test that function returns a numpy.array"""

    output = allantoolkit.utils.phase2frequency(data, rate=RATE)

    assert isinstance(output, np.ndarray)


def test_return_size(data_with_gaps):
    """Test that output phase data has reduced dimensionality due to
    first difference."""

    # using data without gaps to make check easier
    output = allantoolkit.utils.phase2frequency(data_with_gaps, rate=RATE)

    assert output.size == data_with_gaps.size - 1


def test_all_gaps(data_with_only_gaps):
    """"Test we get same (reduced) array if data is full of gaps"""

    output = allantoolkit.utils.phase2frequency(data_with_only_gaps, rate=RATE)

    np.testing.assert_array_equal(output, data_with_only_gaps[:-1])


def test_single_value():
    """Test behaviour if one and only value in dataset"""

    # fill one value
    data = np.array([np.NaN, np.NaN, np.NaN, 3., np.NaN])

    output = allantoolkit.utils.phase2frequency(data, rate=RATE)

    output_theory = np.array([np.NaN, np.NaN, np.NaN, np.NaN])

    np.testing.assert_array_equal(output, output_theory)


def test_two_values():
    """Test behaviour if just enough values in dataset to convert to
    fractional frequency"""

    # fill one value
    data = np.array([np.NaN, np.NaN, np.NaN, 3., 4., np.NaN])

    output = allantoolkit.utils.phase2frequency(data, rate=RATE)

    output_theory = np.array([np.NaN, np.NaN, np.NaN, 1.*RATE, np.NaN])

    np.testing.assert_array_equal(output, output_theory)
