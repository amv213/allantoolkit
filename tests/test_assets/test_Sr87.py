import allantoolkit
import pathlib
import pytest
import numpy as np

# top level directory with asset files for these tests
ASSETS_DIR = pathlib.Path(__file__).parent.parent / 'assets/Sr87'

y_ref = np.loadtxt(ASSETS_DIR / 'plot3.txt')[:, 1]
x_ref = np.loadtxt(ASSETS_DIR / 'plot3_phase.txt')
x0_ref = np.loadtxt(ASSETS_DIR / 'plot3_phase_norm.txt')

# Data sampling rate
RATE = 0.4  # Hz, tau_0 = 2.5s


def test_frequency2phase():
    """Test that fractional frequency data is converted to phase data in the
    same way as Stable32. This will make sure all Stable32 deviations
    calculated on frequency data should match allantoolkit deviations which
    are (mostly) calculated on phase data behind the scenes."""
    output = allantoolkit.utils.frequency2phase(y=y_ref, rate=RATE)

    assert np.allclose(x_ref, output, rtol=1e-12)


# input result files and function which should replicate them
results = [
    ('adev_octave.txt', allantoolkit.allantools.adev),
    ('oadev_octave.txt', allantoolkit.allantools.oadev),
]


@pytest.mark.parametrize('result, fct', results)
def test_generic_octave(result, fct):
    """Test allantoolkit deviations give the same results as Stable32"""

    ref = np.loadtxt(ASSETS_DIR / result)
    expected_taus = ref[:, 1]
    expected_ns = ref[:, 2]
    expected_devs = ref[:, 5]

    output = fct(data=y_ref, rate=RATE, data_type='freq', taus='octave')

    assert np.allclose(expected_taus, output.taus)
    assert np.allclose(expected_ns, output.ns)
    assert np.allclose(expected_devs, output.devs)



