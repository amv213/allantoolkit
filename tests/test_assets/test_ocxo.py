"""
  Test for allantoolkit (https://github.com/aewallin/allantools)
  Stable32 was used to calculate the deviations we compare against.

  AW2015-06-26
  
  The dataset is from the 10 MHz output at the back of an HP Impedance Analyzer
  measured with Keysight 53230A counter, 1.0s gate, RCON mode, with H-maser 10MHz reference

"""

import logging
import pathlib
import pytest
import allantoolkit


# Set testing logger to debug mode
logging.basicConfig()
logging.getLogger('allantoolkit.testutils').setLevel("DEBUG")

# Directory with asset files
ASSETS_DIR = pathlib.Path(__file__).parent.parent / 'assets/ocxo'

# Raw data onto which to perform statistics and check it matches
# (this is the scaled and normalised version of ocxo_frequency)
Y = allantoolkit.utils.read_datafile(ASSETS_DIR / 'ocxo_frequency0.txt')

# Data sampling rate
RATE = 1.  # Hz

# Function to check, test_alpha, test_ci
# FIXME: Test_ci is set to False because CI implementation doesn't match yet
#  Stable32 results.
params = [
    (allantoolkit.allantools.adev, True, False),
    (allantoolkit.allantools.oadev, True, False),
    (allantoolkit.allantools.mdev, True, False),
    (allantoolkit.allantools.tdev, True, False),
    (allantoolkit.allantools.hdev, True, False),
    (allantoolkit.allantools.ohdev, True, False),
    (allantoolkit.allantools.totdev, True, False),
]


@pytest.mark.parametrize('func, test_alpha, test_ci', params)
def test_dev(func, test_alpha, test_ci):
    """Test that Allantoolkit deviation results match the reference Stable32
    results."""

    fn = ASSETS_DIR / (func.__name__ + '_octave_0.txt')

    allantoolkit.testutils.test_Stable32_run(data=Y, func=func, rate=RATE,
                                             data_type='freq',
                                             taus='octave',
                                             fn=fn, test_alpha=test_alpha,
                                             test_ci=test_ci)
