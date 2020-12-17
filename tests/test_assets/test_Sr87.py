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

# Raw frequency data collected from experiment
Y = allantoolkit.testutils.read_datafile(ASSETS_DIR / 'freq/freq_data.txt')

# Phase and normalised phase data, obtained by Stable32 conversion of freq
X = allantoolkit.testutils.read_datafile(ASSETS_DIR / 'phase/phase_data.txt')
X0 = allantoolkit.testutils.read_datafile(ASSETS_DIR /
                                          'phase0/phase0_data.txt')

# Data sampling rate
RATE = 0.4  # Hz, tau_0 = 2.5s


def test_frequency2phase():
    """Test that fractional frequency data is converted to phase data in the
    same way as Stable32. This will make sure all Stable32 deviations
    calculated on frequency data should match allantoolkit deviations which
    are (mostly) calculated on phase data behind the scenes."""
    output = allantoolkit.utils.frequency2phase(y=Y, rate=RATE)

    assert np.allclose(X, output)


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

    datafile = ASSETS_DIR / 'phase0/phase0_data.txt'

    result_fn = fct.__name__ + '.txt'
    resultfile = ASSETS_DIR / 'phase0/octave' / result_fn

    return allantoolkit.testutils.test_row_by_row(fct, datafile, RATE,
                                                  resultfile, tolerance=1e-4)


@pytest.mark.parametrize('fct', fcts)
def test_generic_phase_decade(fct):
    """Test allantoolkit deviations calculated on phase data at octave
    averaging times, give the same results as Stable32 deviations calculated
    on phase data at octave averaging times"""

    datafile = ASSETS_DIR / 'phase0/phase0_data.txt'

    result_fn = fct.__name__ + '.txt'
    resultfile = ASSETS_DIR / 'phase0/decade' / result_fn

    return allantoolkit.testutils.test_row_by_row(fct, datafile, RATE,
                                                  resultfile, tolerance=1e-4)


@pytest.mark.parametrize('fct', fcts)
def test_generic_phase_all(fct):
    """Test allantoolkit deviations calculated on phase data at octave
    averaging times, give the same results as Stable32 deviations calculated
    on phase data at octave averaging times"""

    datafile = ASSETS_DIR / 'phase0/phase0_data.txt'

    result_fn = fct.__name__ + '.txt'
    resultfile = ASSETS_DIR / 'phase0/all' / result_fn

    return allantoolkit.testutils.test_row_by_row(fct, datafile, RATE,
                                                  resultfile, tolerance=1e-4)