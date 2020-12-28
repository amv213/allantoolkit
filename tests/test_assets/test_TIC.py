"""
  Test for allantoolkit (https://github.com/aewallin/allantools)
  Stable32 was used to calculate the deviations we compare against.

  AW2015-03-29
"""

import logging
import pytest
import pathlib
import allantoolkit

# Set testing logger to debug mode
logging.basicConfig()
logging.getLogger('allantoolkit.testutils').setLevel("DEBUG")

# Directory with asset files
ASSETS_DIR = pathlib.Path(__file__).parent.parent / 'assets/Keysight53230A_ti_noise_floor'

# Raw data onto which to perform statistics and check it matches
X = allantoolkit.utils.read_datafile(ASSETS_DIR / 'tic_phase.txt')

# Data sampling rate
RATE = 1.  # Hz

# Function to check
params = [
    # allantoolkit.allantools.adev,  # ADEV file is on many-tau instead of oct
    allantoolkit.devs.oadev,
    allantoolkit.devs.mdev,
    allantoolkit.devs.tdev,
    allantoolkit.devs.hdev,
    allantoolkit.devs.ohdev,
    allantoolkit.devs.totdev,
    allantoolkit.devs.tierms,
]


@pytest.mark.parametrize('func', params)
def test_dev(func):
    """Test that Allantoolkit deviation results match the reference Stable32
    results."""

    fn = ASSETS_DIR / ('tic_' + func.__name__ + '.txt')

    allantoolkit.testutils.test_Stable32_run(data=X, func=func, rate=RATE,
                                             data_type='phase', taus='octave',
                                             fn=fn, test_alpha=True,
                                             test_ci=False)

    # FIXME: Test_ci is set to False because CI implementation doesn't match
    #  yet Stable32 results.
