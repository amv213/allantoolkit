"""
  Test for allantoolkit (https://github.com/aewallin/allantools)
  Stable32 was used to calculate the deviations we compare against.

  The 5071A_phase.txt is a dataset collected with a time-interval-counter
  between 1 pulse-per-second outputs from a 5071A Cs clock against a H-maser
 
  first datapoint          1391174210 2014-01-31 13:16:50 UTC 
  last datapoint           1391731199 2014-02-06 23:59:59 UTC
  556990 datapoints in total

  This test uses log-spaced tau-values (i.e. 1, 2, 4, 10, etc.)
  The CS5071A_test_all.py test much more tau-values (1,2,3,4, etc.) but is slower.

  AW2014-02-07
"""

import pytest
import logging
import pathlib
import allantoolkit

# Set testing logger to debug mode
logging.basicConfig()
logging.getLogger('allantoolkit.testutils').setLevel("DEBUG")

# Directory with asset files
ASSETS_DIR = pathlib.Path(__file__).parent.parent / 'assets/Cs5071A'

# Raw data onto which to perform statistics and check it matches
X = allantoolkit.testutils.read_datafile(ASSETS_DIR / '5071A_phase.txt.gz')

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
    (allantoolkit.allantools.tierms, False, False),
]


@pytest.mark.parametrize('func, test_alpha, test_ci', params)
def test_dev(func, test_alpha, test_ci):
    """Test that Allantoolkit deviation results match the reference Stable32
    results."""

    fn = ASSETS_DIR / (func.__name__ + '_decade.txt')

    allantoolkit.testutils.test_Stable32_run(data=X, func=func, rate=RATE,
                                             data_type='phase', taus='decade',
                                             fn=fn, test_alpha=test_alpha,
                                             test_ci=test_ci)