"""
  Pink frequency noise test for allantoolkit (https://github.com/aewallin/allantools)
  Stable32 was used to calculate the deviations we compare against.

  the pink_frequency.txt was generated with noise.py, for documentation see that file.

"""

import pytest
import pathlib
import allantoolkit.ci
import allantoolkit.allantools as allan
import allantoolkit.testutils as testutils


# top level directory with asset files
ASSETS_DIR = pathlib.Path(__file__).parent.parent / 'assets'

# input data files, and associated verbosity, tolerance, and acquisition rate
assets = [
    ('pink_frequency/pink_frequency.txt', 1, 1e-4, 1/float(42.0)),
]


# input result files and function which should replicate them
results = [
    ('adev.txt', allantoolkit.allantools.adev),
    ('oadev.txt', allantoolkit.allantools.oadev),
    ('mdev.txt', allantoolkit.allantools.mdev),
    ('tdev.txt', allantoolkit.allantools.tdev),
    ('hdev.txt', allantoolkit.allantools.hdev),
    ('ohdev.txt', allantoolkit.allantools.ohdev),
    ('totdev_alpha0.txt', allantoolkit.allantools.totdev),
]


@pytest.mark.slow
@pytest.mark.parametrize('datafile, verbose, tolerance, rate', assets)
@pytest.mark.parametrize('result, fct', results)
def test_generic(datafile, result, fct, verbose, tolerance, rate):

    datafile = ASSETS_DIR / datafile
    result = datafile.parent / result

    testutils.test_row_by_row(fct, datafile, rate, result,
                              verbose=verbose, tolerance=tolerance,
                              frequency=True, normalize=False)


@pytest.mark.slow
@pytest.mark.parametrize('datafile, verbose, tolerance, rate', assets)
def test_noise_id(datafile, verbose, tolerance, rate):
    """ test for noise-identification """

    datafile = ASSETS_DIR / datafile
    result = datafile.parent / 's32_oadev_octave.txt'

    s32_rows = testutils.read_stable32(result, 1.0)
    phase = testutils.read_datafile(datafile)

    for s32 in s32_rows:
        tau, alpha, af = s32['tau'], s32['alpha'], int(s32['m'])
        try:
            alpha_int = allantoolkit.ci.autocorr_noise_id(
                phase, data_type='freq', af=af)[0]

            # if len(phase)/af > 30: # noise-id only works for length 30 or longer time-series
            assert alpha_int == alpha
            print(tau, alpha, alpha_int)
        except NotImplementedError:
            print("no noise-ID: ", tau, alpha, alpha_int)
