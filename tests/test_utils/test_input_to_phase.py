import pytest
import allantoolkit
import numpy as np


def test_invalid_data_type(data):
    """Test that function raises ValueError if data_type is not `phase` or
    `freq`."""

    data_type = 'phrase'

    with pytest.raises(ValueError):
        allantoolkit.utils.input_to_phase(data=data, rate=12.3,
                                          data_type=data_type)


def test_return_same(data):
    """Test that input is preserved if provided phase data."""

    output = allantoolkit.utils.input_to_phase(data=data, rate=12.3,
                                               data_type='phase')
    assert np.array_equal(data, output)


def test_return_type(data):
    """Test that function returns a numpy.array."""

    output = allantoolkit.utils.input_to_phase(data=data, rate=12.3,
                                               data_type='freq')
    assert isinstance(output, np.ndarray)
