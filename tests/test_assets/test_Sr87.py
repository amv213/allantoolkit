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

import logging
import allantoolkit
import pathlib
import pytest
import numpy as np

logging.basicConfig()
logging.getLogger('allantoolkit.testutils').setLevel("DEBUG")

# top level directory with original (frequency) data for these tests
ASSETS_DIR = pathlib.Path(__file__).parent.parent / 'assets/Sr87'

# Raw frequency data collected from experiment
Y = allantoolkit.utils.read_datafile(ASSETS_DIR / 'freq/freq_data.txt')

# Phase and normalised phase data, obtained by Stable32 conversion of freq
X = allantoolkit.utils.read_datafile(ASSETS_DIR / 'phase/phase_data.txt')
X0 = allantoolkit.utils.read_datafile(ASSETS_DIR /
                                          'phase0/phase0_data.txt')

# Data sampling rate
RATE = 0.4  # Hz, tau_0 = 2.5s


# INPUT DATA MANIPULATIONS:

def test_frequency2phase():
    """Test that fractional frequency data is converted to phase data in the
    same way as Stable32. This will make sure all Stable32 deviations
    calculated on frequency data should match allantoolkit deviations which
    are (mostly) calculated on phase data behind the scenes."""
    output = allantoolkit.utils.frequency2phase(y=Y, rate=RATE,
                                                normalize=False)

    assert np.allclose(X, output, atol=1e-32)


def test_frequency2phase0():
    """Test that fractional frequency data is converted to phase data in the
    same way as Stable32. This will make sure all Stable32 deviations
    calculated on frequency data should match allantoolkit deviations which
    are (mostly) calculated on phase data behind the scenes."""
    output = allantoolkit.utils.frequency2phase(y=Y, rate=RATE,
                                                normalize=True)

    assert np.allclose(X0, output, atol=1e-28)
    # Note: Looks like Stable32 normalisation introduces discrepancies at
    # the 1e-29 level


def test_phase2frequency():
    """Test that can losslessly convert fractional frequency data to phase
    data, and back."""

    x = allantoolkit.utils.frequency2phase(y=Y, rate=RATE, normalize=False)

    y = allantoolkit.utils.phase2frequency(x=x, rate=RATE)

    assert np.allclose(Y, y, atol=1e-32)


def test_phase02frequency():
    """Test that can losslessly convert fractional frequency data to
    normalised phase data, and back."""

    x0 = allantoolkit.utils.frequency2phase(y=Y, rate=RATE, normalize=True)

    y = allantoolkit.utils.phase2frequency(x=x0, rate=RATE)

    assert np.allclose(Y, y, atol=1e-32)


# TEST RESULTS

input_data = [
    (X0, 'phase'),
    (Y, 'freq'),
]

tau_types = [
    'octave',
    'decade',
]

fcts = [
    allantoolkit.devs.adev,
    allantoolkit.devs.oadev,
    allantoolkit.devs.mdev,
    allantoolkit.devs.tdev,
    allantoolkit.devs.hdev,
    allantoolkit.devs.ohdev,
    allantoolkit.devs.totdev,
    #pytest.param(allantoolkit.devs.mtotdev,  marks=pytest.mark.slow),
    #pytest.param(allantoolkit.devs.ttotdev, marks=pytest.mark.slow),
    #pytest.param(allantoolkit.devs.htotdev, marks=pytest.mark.slow),
    allantoolkit.devs.theo1,
    #allantoolkit.devs.mtie,  # FIXME: fails if CPU under load
    #allantoolkit.devs.tierms,  # FIXME: fails if CPU under load
]


@pytest.mark.parametrize('data, data_type', input_data)
@pytest.mark.parametrize('taus', tau_types)
@pytest.mark.parametrize('func', fcts)
def test_dev(data, data_type, func, taus):

    if data_type == 'freq':
        fn = ASSETS_DIR / data_type / taus / (func.__name__ + '.txt')
    else:
        fn = ASSETS_DIR / (data_type + '0') / taus / (func.__name__ + '.txt')

    allantoolkit.testutils.test_Stable32_run(data=data, func=func, rate=RATE,
                                             data_type=data_type, taus=taus,
                                             fn=fn, test_alpha=True,
                                             test_ci=False)

# FIXME: Many-taus tests will fail because Stable32 `Run` noise estimation
#  when many-taus is selected starts happening in an undocumented hidden way
#  which is not the same as for other tau types, and also doesn't correspond
#  anymore to what the Stable32 `Sigma` box says the noise should be...
#  either find how Stable32 `Run` is estimating noise for 'many-tau` or just
#  don't bother and don't try to match results with Stable32
@pytest.mark.skip
@pytest.mark.parametrize('data, data_type', input_data)
@pytest.mark.parametrize('func', fcts)
def test_dev_many_taus(data, data_type, func):

    if data_type == 'freq':
        fn = ASSETS_DIR / data_type / 'many' / (func.__name__ + '.txt')
    else:
        fn = ASSETS_DIR / (data_type + '0') / 'many' / (func.__name__ + '.txt')

    allantoolkit.testutils.test_Stable32_run(data=data, func=func, rate=RATE,
                                             data_type=data_type, taus='many',
                                             fn=fn, test_alpha=True,
                                             test_ci=False)


# FIXME: check why doesn't match Stable32
@pytest.mark.skip
@pytest.mark.parametrize('data, data_type', input_data)
def test_fastu_mtie(data, data_type):

    func = allantoolkit.devs.mtie

    if data_type == 'freq':
        fn = ASSETS_DIR / data_type / 'fastu' / (func.__name__ + '.txt')
    else:
        fn = ASSETS_DIR / (data_type + '0') / 'fastu' / (func.__name__ +
                                                         '.txt')

    allantoolkit.testutils.test_Stable32_run(data=data, func=func, rate=RATE,
                                             data_type=data_type, taus='all',
                                             fn=fn, test_alpha=True,
                                             test_ci=False)