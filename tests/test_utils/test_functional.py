import pytest
import allantoolkit
import numpy as np

# Randomise fixed parameters
RATE = np.random.random()*100


@pytest.mark.skip
def test_circular_conversion(data):
    """Test if the round-trip conversion of phase data to frequency and
    back, is invariant."""

    # FIXME: it is probably normal that this is not invariant

    # The last phase point will not be used to make a frequency value
    y = allantoolkit.utils.phase2frequency(data, rate=RATE)

    # This will introduce an arbitrary phase point at the beginning
    x = allantoolkit.utils.frequency2phase(y, rate=RATE)

    assert np.array_equal(x, data)


def test_circular_conversion_size(data):
    """Test that the round-trip conversion of phase data to frequency and
    back, preservers the data shape if no trailing or leading gaps"""

    # The last phase point will not be used to make a frequency value
    y = allantoolkit.utils.phase2frequency(data, rate=RATE)

    # This will introduce an arbitrary phase point at the beginning
    x = allantoolkit.utils.frequency2phase(y, rate=RATE)

    assert data.size == x.size


def test_circular_conversion_size_with_gaps(data_with_gaps):
    """Test that the round-trip conversion of phase data with leading and
    trailing gaps to frequency and back, leads to correct dimensions."""

    # Artificial leading and trailing gaps
    # Making wall double sized or first conversion to frequency may expose
    # additional trailing/leading NaNs which would falsify results
    data_with_gaps[0], data_with_gaps[1], data_with_gaps[2], data_with_gaps[
        3] = np.NaN, np.NaN, 1, 1
    data_with_gaps[-1], data_with_gaps[-2], data_with_gaps[-3] = np.NaN, 1, 1

    # The last phase point will not be used to make a frequency value
    y = allantoolkit.utils.phase2frequency(data_with_gaps, rate=RATE)

    # This will trim leading and trailing gaps, and introduce an arbitrary
    # phase point at the beginning
    x = allantoolkit.utils.frequency2phase(y, rate=RATE)

    assert x.size == (data_with_gaps.size - 1) + 1 - 3
