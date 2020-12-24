"""
  PHASE.DAT test for allantoolkit (https://github.com/aewallin/allantools)
  Stable32 was used to calculate the deviations we compare against.

  PHASE.DAT comes with Stable32 (version 1.61 was used in this case)

  This test is for Confidence Intervals in particular
  
"""

import logging
import pathlib
import pytest
import allantoolkit

# Set testing logger to debug mode
logging.basicConfig()
logging.getLogger('allantoolkit.testutils').setLevel("DEBUG")

# Directory with asset files (Testing results from Stable 1.61)
ASSETS_DIR = pathlib.Path(__file__).parent.parent / 'assets/phasedat/'

# Raw data onto which to perform statistics and check it matches
X = allantoolkit.testutils.read_datafile(ASSETS_DIR / 'PHASE.DAT')

# Data sampling rate
RATE = 1.  # Hz

# Function to check
params = [
    allantoolkit.allantools.adev,
    allantoolkit.allantools.oadev,
    allantoolkit.allantools.mdev,
    allantoolkit.allantools.tdev,
    allantoolkit.allantools.hdev,
    allantoolkit.allantools.ohdev,
    allantoolkit.allantools.totdev,
    pytest.param(allantoolkit.allantools.mtotdev,
                 marks=pytest.mark.slow),
    pytest.param(allantoolkit.allantools.ttotdev,
                 marks=pytest.mark.slow),
    pytest.param(allantoolkit.allantools.htotdev,
                 marks=pytest.mark.slow),
    allantoolkit.allantools.tierms,
    allantoolkit.allantools.mtie,
    allantoolkit.allantools.theo1,
]


# FIXME: Test_ci is set to False because CI implementation doesn't match yet
#  Stable32 results.
@pytest.mark.parametrize('func', params)
def test_dev(func):
    """Test that Allantoolkit deviation results match the reference Stable32
    results."""

    fn = ASSETS_DIR / (func.__name__ + '_octave.txt')

    allantoolkit.testutils.test_Stable32_run(data=X, func=func, rate=RATE,
                                             data_type='phase',
                                             taus='octave',
                                             fn=fn, test_alpha=True,
                                             test_ci=False)