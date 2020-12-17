"""Test data based on a Strontium Optical Lattice Clock. Reference results
were generated with Stable32 on the data in plot3.txt

Data type: fractional frequency
Sampling rate: 0.4 Hz
Gaps: No

This suite of test is useful to check that Allantoolkit works in the same
way as Stable32 also on data acquired as fractional frequency. This suite of
test will not check if gaps are handled in the same way, as the dataset has
no gaps.
"""

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
    ('mdev_octave.txt', allantoolkit.allantools.mdev),
    ('tdev_octave.txt', allantoolkit.allantools.tdev),
    ('hdev_octave.txt', allantoolkit.allantools.hdev),
    ('ohdev_octave.txt', allantoolkit.allantools.ohdev),
    ('totdev_octave.txt', allantoolkit.allantools.totdev),
    pytest.param('mtotdev_octave.txt', allantoolkit.allantools.mtotdev,
                 marks=pytest.mark.slow),
]


# FIXME: add tests of noise type identified, and upper and minimum error
#  bounds once implemented those calculations in allantoolkit
@pytest.mark.parametrize('result, fct', results)
def test_generic_octave(result, fct):
    """Test allantoolkit deviations give the same results as Stable32"""

    datafile = ASSETS_DIR / 'plot3.txt'
    result = datafile.parent / result

    s32rows = allantoolkit.testutils.read_stable32(resultfile=result,
                                                   datarate=RATE)

    for row in s32rows:

        data = allantoolkit.testutils.read_datafile(datafile)

        (taus, devs, errs_lo, errs_hi, ns) = fct(data=data, rate=RATE,
                                                 data_type='freq',
                                                 taus=row['tau'])

        print("n check: ", allantoolkit.testutils.check_equal(ns[0], row['n']))
        print("dev check: ", allantoolkit.testutils.check_approx_equal(
            devs[0], row['dev']))


        #print("min dev check: ",  lo, row['dev_min'],
        # testutils.check_approx_equal( lo, row['dev_min'], tolerance=1e-3 ) )
        #print("max dev check: ", hi, row['dev_max'],
        # testutils.check_approx_equal( hi, row['dev_max'], tolerance=1e-3 ) )


