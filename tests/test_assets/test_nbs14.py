"""
  NBS14 test for allantoolkit (https://github.com/aewallin/allantools)

  nbs14 datasets are from http://www.ieee-uffc.org/frequency-control/learning-riley.asp
  
  Stable32 was used to calculate the deviations we compare against.

  The small dataset and deviations are from
  http://www.ieee-uffc.org/frequency-control/learning-riley.asp
  http://www.wriley.com/paper1ht.htm
  
  see also:
  NIST Special Publication 1065
  Handbook of Frequency Stability Analysis
  http://tf.nist.gov/general/pdf/2220.pdf
  around page 107
"""

import pytest
import pathlib
import logging
import allantoolkit
import numpy as np

logging.basicConfig()
logging.getLogger('allantoolkit.testutils').setLevel("DEBUG")

# top level directory with original (frequency) data for these tests
ASSETS_DIR = pathlib.Path(__file__).parent.parent / 'assets/nbs14'

# Raw frequency test frequency data
Y10 = allantoolkit.utils.read_datafile(ASSETS_DIR / 'nbs_10.txt')
Y1000 = allantoolkit.utils.read_datafile(ASSETS_DIR / 'nbs_1000.txt')

# Data sampling rate
RATE = 1.  # Hz

# Euivalent phase data
X10 = allantoolkit.utils.frequency2phase(y=Y10, rate=RATE)
X1000 = allantoolkit.utils.frequency2phase(y=Y1000, rate=RATE)

# NBS 10 TEST

# Expected reference results; from http://www.wriley.com/paper1ht.htm#Table_II
nbs14_10_ref = {
    # m = 1, 2
    'adev': [91.22945, 115.8082],
    'oadev': [91.22945, 85.95287],
    'mdev': [91.22945, 74.78849],
    # 'totdev': [91.22945, 98.31100],
    # Correction from http://tf.nist.gov/general/pdf/2220.pdf page 107
    'totdev': [91.22945, 93.90379],
    'hdev': [70.80608, 116.7980],
    'tdev': [52.67135, 86.35831],
}


fcts10 = [
    allantoolkit.allantools.adev,
    allantoolkit.allantools.oadev,
    allantoolkit.allantools.mdev,
    allantoolkit.allantools.totdev,
    allantoolkit.allantools.hdev,
    allantoolkit.allantools.tdev,
]


input_data = [
    (Y10, 'freq'),
    (X10, 'phase'),
]

@pytest.mark.parametrize('data, data_type', input_data)
@pytest.mark.parametrize('func', fcts10)
def test_nbs10(data, data_type, func):

    expected_devs = nbs14_10_ref[func.__name__]

    out = func(data=data, rate=RATE, data_type=data_type,
               taus=np.array([1, 2]))

    print(func.__name__)
    print(expected_devs)
    print(out.afs)
    print(out.devs)

    # Check deviations are the same
    for i, dev in enumerate(expected_devs):
        assert np.format_float_scientific(dev, 5) == \
               np.format_float_scientific(out.devs[i], 5)


# NBS 1000 TEST


# Expected reference results; from http://www.wriley.com/paper1ht.htm#Table_III
nbs14_1000_ref = {
    # m = 1, 10, 100
    'adev': [2.922319e-01, 9.965736e-02, 3.897804e-02],
    'oadev': [2.922319e-01, 9.159953e-02, 3.241343e-02],
    'mdev': [2.922319e-01, 6.172376e-02, 2.170921e-02],
    'tdev': [1.687202e-01, 3.563623e-01, 1.253382e-00],
    'hdev': [2.943883e-01, 1.052754e-01, 3.910860e-02],
    'ohdev': [2.943883e-01, 9.581083e-02, 3.237638e-02],
    'totdev': [2.922319e-01, 9.134743e-02, 3.406530e-02],
    # 'mtotdev': [2.418528e-01, 6.499161e-02, 2.287774e-02],
    # But Stable32 doesn't apply bias correction so need to use this:
    'mtotdev': [2.06639e-01, 5.55289e-02, 1.95468e-02],
    # 'ttotdev': [1.396338e-01, 3.752293e-01, 1.320847e+00],
    # But Stable32 doesn't apply bias correction so need to use this:
    'ttotdev': [1.19303e-01, 3.20596e-01, 1.12853e+00],
    'htotdev': [2.943883e-01, 9.614787e-02, 3.058103e-02],
}

fcts1000 = [
    allantoolkit.allantools.adev,
    allantoolkit.allantools.oadev,
    allantoolkit.allantools.mdev,
    allantoolkit.allantools.tdev,
    allantoolkit.allantools.hdev,
    allantoolkit.allantools.ohdev,
    allantoolkit.allantools.totdev,
    pytest.param(allantoolkit.allantools.mtotdev,  marks=pytest.mark.slow),
    pytest.param(allantoolkit.allantools.ttotdev, marks=pytest.mark.slow),
    pytest.param(allantoolkit.allantools.htotdev, marks=pytest.mark.slow),
]


input_data = [
    (X1000, 'phase'),
    (Y1000, 'freq'),
]


@pytest.mark.parametrize('data, data_type', input_data)
@pytest.mark.parametrize('func', fcts1000)
def test_nbs1000(data, data_type, func):

    expected_devs = nbs14_1000_ref[func.__name__]

    out = func(data=data, rate=RATE, data_type=data_type,
               taus=np.array([1, 10, 100]))

    # Check deviations are the same
    for i, dev in enumerate(expected_devs):
        assert np.format_float_scientific(dev, 5) == \
               np.format_float_scientific(out.devs[i], 5)
