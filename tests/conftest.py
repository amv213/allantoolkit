import pytest
import allantoolkit
import numpy as np


@pytest.fixture
def data():
    """An all purpose dataset"""
    return allantoolkit.noise.white(100)


@pytest.fixture
def data_with_gaps(data):
    """An all purpose dataset with missing values"""

    # Set the number of values to replace
    prop = np.random.random()
    num_rep = int(data.size * prop)

    # randomly choose indices of the data array
    i = [np.random.choice(range(data.size)) for _ in range(num_rep)]

    # Change values with NaNs
    data[i] = np.NaN

    return data


@pytest.fixture
def data_with_only_gaps(data):

    return np.full_like(data, np.NaN)


@pytest.fixture
def dataset():
    return allantoolkit.dataset.Dataset(allantoolkit.noise.white(100))


@pytest.fixture
def noisegen():
    return allantoolkit.noise_kasdin.Noise()
