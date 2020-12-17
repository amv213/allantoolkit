"""
  Test for allantoolkit (https://github.com/aewallin/allantools)
  Stable32 was used to calculate the deviations we compare against.

  AW2015-03-29
"""

import pytest
import pathlib
import allantoolkit
import allantoolkit.testutils as testutils


# top level directory with asset files
ASSETS_DIR = pathlib.Path(__file__).parent.parent / 'assets'

# input data files, and associated verbosity, tolerance, and acquisition rate
assets = [
    ('Keysight53230A_ti_noise_floor/tic_phase.txt', True, 1e-4, 1.),
]

# input result files and function which should replicate them
results = [
    ('tic_adev.txt', allantoolkit.allantools.adev),
    ('tic_oadev.txt', allantoolkit.allantools.oadev),
    ('tic_mdev.txt', allantoolkit.allantools.mdev),
    ('tic_tdev.txt', allantoolkit.allantools.tdev),
    ('tic_hdev.txt', allantoolkit.allantools.hdev),
    ('tic_ohdev.txt', allantoolkit.allantools.ohdev),
    ('tic_totdev.txt', allantoolkit.allantools.totdev),
    ('tic_tierms.txt', allantoolkit.allantools.tierms),
]


@pytest.mark.slow
@pytest.mark.parametrize('datafile, verbose, tolerance, rate', assets)
@pytest.mark.parametrize('result, fct', results)
def test_generic(datafile, result, fct, verbose, tolerance, rate):

    datafile = ASSETS_DIR / datafile
    result = datafile.parent / result

    testutils.test_row_by_row(fct, datafile, rate, result,
                              verbose=verbose, tolerance=tolerance)


@pytest.mark.parametrize('datafile, verbose, tolerance, rate', assets)
def test_noise_id(datafile, verbose, tolerance, rate):
    """ test for noise-identification """

    datafile = ASSETS_DIR / datafile
    result = datafile.parent / 'tic_oadev.txt'

    s32_rows = testutils.read_stable32(result)
    phase = testutils.read_datafile(datafile)

    for s32 in s32_rows:

        tau, alpha, af = s32[1], s32[3], int(s32[0])

        try:
            alpha_int = allantoolkit.ci.autocorr_noise_id(
                phase, af=af)[0]

            # if len(phase)/af > 30: # noise-id only works for length 30
            # or longer time-series
            assert alpha_int == alpha
            print(tau, alpha, alpha_int)

        except NotImplementedError:
            print("no noise-ID: ", tau, alpha, alpha_int)
