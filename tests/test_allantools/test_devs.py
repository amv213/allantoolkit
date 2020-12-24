import allantoolkit
import numpy as np
import pytest


N = 128
RATE = 1.
d = allantoolkit.noise.white(N)
d = allantoolkit.utils.frequency2phase(y=d, rate=RATE)

expected_all = np.arange(1, 43)
# adev Stable32 stop-ratio is at 25
expected_octave = [1.,   2.,   4.,   8.,  16.]
expected_decade = [1., 2., 4., 10., 20.]
expected_reduced_10 = [1., 2., 3., 4., 5., 7., 8., 11., 14., 17.,
                       22., 28., 35.]
expected_reduced_2 = [1., 5., 17.]


def test_tau_generator_empty():
    out = allantoolkit.allantools.adev(d)
    np.testing.assert_allclose(out.taus, expected_octave)

# TODO: Revisit what behaviour should be if passing an empty string
@pytest.mark.skip
def test_tau_generator_empty_list():
    out = allantoolkit.allantools.adev(d, taus=[])
    np.testing.assert_allclose(out.taus, expected_octave)


def test_tau_generator_all():
    out = allantoolkit.allantools.adev(d, rate=RATE, taus="all")
    np.testing.assert_allclose(out.taus, expected_all)


def test_tau_generator_octave():
    out = allantoolkit.allantools.adev(d, rate=RATE, taus="octave")
    np.testing.assert_allclose(out.taus, expected_octave)


def test_tau_generator_decade():
    out = allantoolkit.allantools.adev(d, rate=RATE, taus="decade")
    np.testing.assert_allclose(out.taus, expected_decade)


def test_tau_generator_1234():
    wanted_taus = [1, 2, 3, 4]
    out = allantoolkit.allantools.adev(d, rate=RATE, taus=wanted_taus)
    np.testing.assert_allclose(out.taus, wanted_taus)


def test_tau_generator_numpy1234():
    wanted_taus = np.array([1, 2, 3, 4])
    out = allantoolkit.allantools.adev(d, rate=RATE, taus=wanted_taus)
    np.testing.assert_allclose(out.taus, wanted_taus)


def test_zero_rate():
    with pytest.raises(ZeroDivisionError):
        allantoolkit.allantools.adev(d, rate=0.0)
