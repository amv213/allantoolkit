"""
  Test for allantoolkit (https://github.com/aewallin/allantools)
  Stable32 was used to calculate the deviations we compare against.

  GPS tests, AW2016-03-17
"""

import logging
import pathlib
import pytest
import allantoolkit


# Set testing logger to debug mode
logging.basicConfig()
logging.getLogger('allantoolkit.testutils').setLevel("DEBUG")

# Directory with asset files
ASSETS_DIR = pathlib.Path(__file__).parent.parent / 'assets/gps'

# Raw data onto which to perform statistics and check it matches
X = allantoolkit.testutils.read_datafile(ASSETS_DIR / 'gps_1pps_phase_data.txt.gz')

# Data sampling rate
RATE = 1.  # Hz

# Function to check
params = [
    # allantoolkit.allantools.adev,  # adevs is decade not octave
    allantoolkit.allantools.oadev,
    allantoolkit.allantools.mdev,
    allantoolkit.allantools.tdev,
    allantoolkit.allantools.hdev,
    allantoolkit.allantools.ohdev,
    allantoolkit.allantools.totdev,
]


@pytest.mark.parametrize('func', params)
def test_dev(func):
    """Test that Allantoolkit deviation results match the reference Stable32
    results."""

    fn = ASSETS_DIR / ('stable32_' + func.__name__.upper() + '_octave.txt')

    allantoolkit.testutils.test_Stable32_run(data=X, func=func, rate=RATE,
                                             data_type='phase', taus='octave',
                                             fn=fn, test_alpha=True,
                                             test_ci=False)

    # FIXME: Test_ci is set to False because CI implementation doesn't match
    #  yet Stable32 results.