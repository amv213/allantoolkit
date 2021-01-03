"""
unit-tests for Kasdin & Walter noise-generator
"""

import pytest
import allantoolkit
import numpy as np
from typing import Tuple


def test_noise():
    N = 500
    # rate = 1.0
    w = allantoolkit.noise.white(N)
    b = allantoolkit.noise.brown(N)
    v = allantoolkit.noise.violet(N)
    p = allantoolkit.noise.pink(N)

    # check output length
    assert w.n == N
    assert b.n == N
    assert v.n == N
    assert p.n == N
    # check output type
    for x in [w, b, v, p]:
        assert type(x.data) == np.ndarray, "%s is not numpy.ndarray" % (
            type(x.data))


@pytest.mark.parametrize("n", range(2, 20))
def test_timeseries_length(n):
    """
        check that the time-series is of correct length
    """
    nr = pow(2, n)

    noise = allantoolkit.noise.white(n=nr)

    assert len(noise.data) == nr


# failing cases (b, tau, qd)
# these seem problematic, even with nr=pow(2,16) and N_averages=30
failing = [(-4, 1, 6e-09),
           (-3, 1, 6e-09),
           (-4, 1, 5e-10),
           (-3, 1, 5e-10),
           (-4, 1, 3e-15),
           (-3, 1, 3e-15),
           ]


@pytest.mark.xfail
@pytest.mark.parametrize("params", failing)
def test_adev_average_failing(noisegen, params: Tuple[int, int, float]):
    test_adev_average(noisegen, params[0], params[1], params[2])


@pytest.mark.parametrize("b", [0, -1, -2, ])
@pytest.mark.parametrize("m", [1, 2, 3, 4, 5, 20, 30])
@pytest.mark.parametrize("qd", [3e-15, 5e-10, 6e-9, ])  # 7e-6 2e-20
def test_adev_average(b, m, qd, nr=pow(2, 16), N_averages=30, rtol=1e-1):
    """
    check that time-series has the ADEV that we expect
    generate many time-series (N_averages)
    compute average adev from the multiple time-series
    """

    rate = 1

    adevs = []
    for n in range(N_averages):

        noise = allantoolkit.noise.custom(beta=b, n=nr, qd=qd, rate=rate)

        out = allantoolkit.devs.adev(noise.data, taus=np.array([m/rate]),
                                     rate=rate)

        adev_calculated = out.devs[0]
        adevs.append(adev_calculated)

    adev_mu = np.mean(adevs)
    adev_predicted = noise.adev(m=m)

    print(b, m, qd, adev_predicted, adev_mu, adev_mu/adev_predicted)

    # NOTE relative tolerance here!
    assert np.isclose(adev_predicted, adev_mu, rtol=rtol, atol=0)


@pytest.mark.parametrize("b", [0, -1, -2, -3, -4])
@pytest.mark.parametrize("m", [1, 2, 3, 4, 5, 20, 30])
@pytest.mark.parametrize("qd", [3e-15, 5e-10, 6e-9, ])  # 7e-6 2e-20
def test_adev(b, m, qd):
    """
    check that time-series has the ADEV that we expect
    """

    rate = 1

    noise = allantoolkit.noise.custom(beta=b, n=pow(2, 16), qd=qd, rate=rate)

    out = allantoolkit.devs.adev(noise.data, taus=np.array([m/rate]), rate=1)
    
    adev_calculated = out.devs[0]
    adev_predicted = noise.adev(m=m)

    print(b, m, qd, adev_calculated, adev_predicted,
          adev_calculated/adev_predicted)

    # NOTE high relative tolerence here !!
    assert np.isclose(adev_calculated, adev_predicted, rtol=3e-1, atol=0)


@pytest.mark.xfail
@pytest.mark.parametrize("b", [-2, -3, -4])
@pytest.mark.parametrize("tau", [1, 3])
@pytest.mark.parametrize("qd", [6e-9, 7e-6])
def test_mdev_failing(noisegen, b, tau, qd):
    test_mdev(noisegen, b, tau, qd)


@pytest.mark.parametrize("b", [0, -1, ])
@pytest.mark.parametrize("m", [2, 4, 5, 20, 30])
@pytest.mark.parametrize("qd", [2e-20])
def test_mdev(b, m, qd):
    """
    check that time-series has the MDEV that we expect
    """

    rate = 1
    noise = allantoolkit.noise.custom(beta=b, n=pow(2, 16), qd=qd, rate=rate)

    out = allantoolkit.devs.mdev(noise.data, taus=np.array([m/rate]), rate=1)

    mdev_calculated = out.devs[0]
    mdev_predicted = noise.mdev(m=m)

    print(b, m, qd, mdev_calculated, mdev_predicted,
          mdev_calculated/mdev_predicted)
    # NOTE high relative tolarence here !!
    assert np.isclose(mdev_calculated, mdev_predicted, rtol=2e-1, atol=0)

