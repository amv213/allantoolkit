import allantoolkit
import numpy as np

N = 128
d = np.random.random(N)
r = 1.0

expected_all = np.arange(1, 43)
# oadev stable32 stop-ratio is at 32:
expected_octave = [1.,   2.,   4.,   8.,  16.,  32.]
expected_decade = [1., 2., 4., 10., 20.]

ab = allantoolkit.noise.white(N).data
ab = allantoolkit.utils.frequency2phase(y=ab, rate=r)
bc = allantoolkit.noise.white(N).data
bc = allantoolkit.utils.frequency2phase(y=bc, rate=r)
ca = allantoolkit.noise.white(N).data
ca = allantoolkit.utils.frequency2phase(y=ca, rate=r)


def test_3ch_1():

    (t, d, e_lo, e_hi, n) = allantoolkit.utils.three_cornered_hat_phase(
        ab, bc, ca, rate=r, taus='decade', dev_type='oadev')

    np.testing.assert_array_equal(t, expected_decade)
