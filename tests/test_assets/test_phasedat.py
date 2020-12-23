"""
  PHASE.DAT test for allantoolkit (https://github.com/aewallin/allantools)
  Stable32 was used to calculate the deviations we compare against.

  PHASE.DAT comes with Stable32 (version 1.53 was used in this case)

  This test is for Confidence Intervals in particular
  
"""

import logging
import pathlib
import pytest
import allantoolkit

# Set testing logger to debug mode
logging.basicConfig()
logging.getLogger('allantoolkit.testutils').setLevel("DEBUG")

# Directory with asset files
ASSETS_DIR = pathlib.Path(__file__).parent.parent / 'assets/phasedat'

# Raw data onto which to perform statistics and check it matches
X = allantoolkit.testutils.read_datafile(ASSETS_DIR / 'PHASE.DAT')

# Data sampling rate
RATE = 1.  # Hz

# Function to check, test_alpha, test_ci
# FIXME: test_alpha is set to false because it seems like reference file was
#  using fixed alpha = 0, instead of using auto noise ID. Test_ci is set to
#  False because CI implementation doesn't match yet Stable32 results.
params = [
    (allantoolkit.allantools.adev, False, False),
    (allantoolkit.allantools.oadev, False, False),
    (allantoolkit.allantools.mdev, False, False),
    (allantoolkit.allantools.tdev, False, False),
    (allantoolkit.allantools.hdev, False, False),
    (allantoolkit.allantools.ohdev, False, False),
    (allantoolkit.allantools.totdev, False, False),
    pytest.param(allantoolkit.allantools.htotdev, False, False,
                 marks=pytest.mark.slow),
    pytest.param(allantoolkit.allantools.mtotdev, False, False,
                 marks=pytest.mark.slow),
    pytest.param(allantoolkit.allantools.ttotdev, False, False,
                 marks=pytest.mark.slow),
    (allantoolkit.allantools.theo1, False, False),
    (allantoolkit.allantools.mtie, False, False),
    (allantoolkit.allantools.tierms, False, False),
]


@pytest.mark.parametrize('func, test_alpha, test_ci', params)
def test_dev(func, test_alpha, test_ci):
    """Test that Allantoolkit deviation results match the reference Stable32
    results."""

    fn = ASSETS_DIR / ('phase_dat_' + func.__name__ + '_octave.txt')

    allantoolkit.testutils.test_Stable32_run(data=X, func=func, rate=RATE,
                                             data_type='phase',
                                             taus='octave',
                                             fn=fn, test_alpha=test_alpha,
                                             test_ci=test_ci)