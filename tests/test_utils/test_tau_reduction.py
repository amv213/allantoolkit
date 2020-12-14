import allantoolkit
import numpy as np
import pytest


# Randomise fixed parameters
RATE = np.random.random()*100

N = 128
d = np.random.random(N)
r = 1.0

expected_all = np.arange(1, 43)
expected_reduced_10 = [1., 2., 3., 4., 5., 7., 8., 11., 14., 17.,
                       22., 28., 35.]
expected_reduced_2 = [1., 5., 17.]


def test_return_type():
    """Test that function returns a `Taus` NamedTuple"""

    output = allantoolkit.utils.tau_reduction(afs=expected_all, rate=r,
                                              n_per_decade=10)

    assert isinstance(output, allantoolkit.utils.Taus)


def test_tau_reduction_10():
    (ms, taus) = allantoolkit.utils.tau_reduction(afs=expected_all, rate=r,
                                                  n_per_decade=10)
    np.testing.assert_allclose(expected_reduced_10, ms)


def test_tau_reduction_2():
    (ms, taus) = allantoolkit.utils.tau_reduction(afs=expected_all, rate=r,
                                                  n_per_decade=2)

    np.testing.assert_allclose(expected_reduced_2, ms)
