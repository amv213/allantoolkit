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

# top level directory with original (frequency) data for these tests
ASSETS_DIR = pathlib.Path(__file__).parent.parent / 'assets/Sr87'
y_ref = np.loadtxt(ASSETS_DIR / 'freq_data_ori.txt', skiprows=10)[:, 1]

# Directory with (un-normalised) phase data, obtained by Stable32 conversion
PHASE_ASSETS_DIR = ASSETS_DIR / 'phase'
x_ref = np.loadtxt(PHASE_ASSETS_DIR / 'phase_data.txt', skiprows=10)

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
fcts = [
    allantoolkit.allantools.adev,
    allantoolkit.allantools.oadev,
    allantoolkit.allantools.mdev,
    allantoolkit.allantools.tdev,
    allantoolkit.allantools.hdev,
    allantoolkit.allantools.ohdev,
    allantoolkit.allantools.totdev,
    pytest.param(allantoolkit.allantools.mtotdev, marks=pytest.mark.slow),
    pytest.param(allantoolkit.allantools.ttotdev, marks=pytest.mark.slow),
    pytest.param(allantoolkit.allantools.htotdev, marks=pytest.mark.slow),
    allantoolkit.allantools.theo1,
    pytest.param(allantoolkit.allantools.mtie, marks=pytest.mark.slow),
    allantoolkit.allantools.tierms,
]


# FIXME: add tests of noise type identified, and upper and minimum error
#  bounds once implemented those calculations in allantoolkit
@pytest.mark.parametrize('fct', fcts)
def test_generic_phase_octave(fct):
    """Test allantoolkit deviations calculated on phase data at octave
    averaging times, give the same results as Stable32 deviations calculated
    on phase data at octave averaging times"""

    datafile = PHASE_ASSETS_DIR / 'phase_data.txt'
    x = allantoolkit.testutils.read_datafile(datafile)

    result_filename = fct.__name__ + '.txt'
    stable32file = datafile.parent / 'octave' / result_filename
    s32rows = allantoolkit.testutils.read_stable32(resultfile=stable32file,
                                                   datarate=RATE)

    for row in s32rows:

        print(f"Ref results: {row}")

        # If checking a theo1, the file will have an effective tau 75% of the
        # original one
        tau_ori = row['tau'] if fct.__name__ != 'theo1' else row['tau'] / 0.75

        (taus, devs, errs_lo, errs_hi, ns) = fct(data=x, rate=RATE,
                                                 taus=tau_ori)

        allantoolkit.testutils.check_equal(ns[0], row['n'])
        allantoolkit.testutils.check_approx_equal(devs[0], row['dev'])

        #print("min dev check: ",  lo, row['dev_min'],
        # testutils.check_approx_equal( lo, row['dev_min'], tolerance=1e-3 ) )
        #print("max dev check: ", hi, row['dev_max'],
        # testutils.check_approx_equal( hi, row['dev_max'], tolerance=1e-3 ) )


