import allantoolkit
import numpy as np

# Randomise fixed parameters
RATE = np.random.random()*100


def test_return_type(data):
    """Test that function returns a numpy.array."""

    output = allantoolkit.utils.frequency2phase(data, rate=RATE)

    assert isinstance(output, np.ndarray)


def test_return_size(data):
    """Test that output phase data has increased dimensionality due to
    integration."""

    output = allantoolkit.utils.frequency2phase(data, rate=RATE)

    assert output.size == data.size + 1
