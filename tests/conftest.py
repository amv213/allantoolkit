import pytest
import allantoolkit


@pytest.fixture
def dataset():
    return allantoolkit.dataset.Dataset(allantoolkit.noise.white(10))


@pytest.fixture
def noisegen():
    return allantoolkit.noise_kasdin.Noise()
