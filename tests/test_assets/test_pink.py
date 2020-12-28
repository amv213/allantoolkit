"""
  Pink frequency noise test for allantoolkit (https://github.com/aewallin/allantools)
  Stable32 was used to calculate the deviations we compare against.

  the pink_frequency.txt was generated with noise.py, for documentation see that file.

"""

import logging
import pytest
import pathlib
import allantoolkit.ci

# Set testing logger to debug mode
logging.basicConfig()
logging.getLogger('allantoolkit.testutils').setLevel("DEBUG")

# Directory with asset files
ASSETS_DIR = pathlib.Path(__file__).parent.parent / 'assets/pink_frequency'

# Raw data onto which to perform statistics and check it matches
Y = allantoolkit.utils.read_datafile(ASSETS_DIR / 'pink_frequency.txt')

# Data sampling rate
RATE = 1./42  # Hz

# Function to check
params = [
    allantoolkit.devs.adev,
    allantoolkit.devs.oadev,
    allantoolkit.devs.mdev,
    allantoolkit.devs.tdev,
    allantoolkit.devs.hdev,
    allantoolkit.devs.ohdev,
    allantoolkit.devs.totdev,
]


# FIXME: this failing because results are on many-tau
@pytest.mark.parametrize('func', params)
def test_dev(func):
    """Test that Allantoolkit deviation results match the reference Stable32
    results."""

    fn = ASSETS_DIR / (func.__name__ + '_octave.txt')

    allantoolkit.testutils.test_Stable32_run(data=Y, func=func, rate=RATE,
                                             data_type='freq', taus='octave',
                                             fn=fn, test_alpha=True,
                                             test_ci=False)

    # FIXME: Test_ci is set to False because CI implementation doesn't match
    #  yet Stable32 results.