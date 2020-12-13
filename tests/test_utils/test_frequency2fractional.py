import pytest
import allantoolkit
import numpy as np

# Randomise fixed parameters
V0 = np.random.random()*100

# Parametrise so that we can run tests both with inferred and given v0
params = [
    None,
    V0,
]


@pytest.mark.parametrize('v0', params)
def test_return_type(data, v0):
    """Test that function returns a numpy.array"""

    output = allantoolkit.utils.frequency2fractional(data, v0=V0)

    assert isinstance(output, np.ndarray)


@pytest.mark.parametrize('v0', params)
def test_return_size(data, v0):
    """Test that function preserves data size"""

    output = allantoolkit.utils.frequency2fractional(data, v0=V0)

    assert output.size == data.size


@pytest.mark.parametrize('v0', params)
def test_all_gaps(data_with_only_gaps, v0):
    """Test that conversion of data with all gaps is invariant."""

    output = allantoolkit.utils.frequency2fractional(data_with_only_gaps,
                                                     v0=v0)

    assert np.array_equal(output, data_with_only_gaps, equal_nan=True)


@pytest.mark.parametrize('v0', params)
def test_empty_data(v0):
    """Test that conversion of empty data is invariant."""

    data = np.array([])

    output = allantoolkit.utils.frequency2fractional(data, v0=v0)

    assert np.array_equal(output, data)


@pytest.mark.parametrize('v0', params)
def test_expected_output(v0):
    """Test conversion to fractional frequency gives expected output"""

    data = np.array([np.NaN, 3, 4, 5, np.NaN, 12])
    v0 = 6. if v0 is None else v0

    output = allantoolkit.utils.frequency2fractional(data, v0=v0)

    expected_output = (data - v0) / v0

    assert np.array_equal(output, expected_output, equal_nan=True)