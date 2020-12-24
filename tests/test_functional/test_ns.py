import logging
import pytest
import allantoolkit
import numpy

# Set testing logger to debug mode
logging.basicConfig()
logging.getLogger('allantoolkit.testutils').setLevel("DEBUG")

N = 500
RATE = 1.
Y = allantoolkit.noise.white(N)
# this test asks for results at unreasonable tau-values
# either zero, not an integer multiple of the data-interval
# or too large, given the length of the dataset
TAUS = [x for x in numpy.logspace(0, 4, 4000)]

funcs = [
    allantoolkit.allantools.adev,
    allantoolkit.allantools.oadev,
    allantoolkit.allantools.mdev,
    allantoolkit.allantools.tdev,
    allantoolkit.allantools.hdev,
    allantoolkit.allantools.ohdev,
    allantoolkit.allantools.totdev,
    allantoolkit.allantools.mtie,
    allantoolkit.allantools.tierms,
]


@pytest.mark.parametrize('func', funcs)
def test_output_shapes(func):

    out = func(data=Y, rate=RATE, taus=TAUS, data_type='freq')

    taus2 = out.taus
    assert(len(taus2) == len(out.devs))
    assert(len(taus2) == len(out.devs_lo))
    assert(len(taus2) == len(out.devs_hi))
    assert(len(taus2) == len(out.ns))

    for n in out.ns:

        if n <= 1:
            print("test of ", func.__name__, " failed: ", n)

        assert(n > 1)  # n should be 2 or more for each tau

    print("test_ns of function ", func.__name__, " OK.")

