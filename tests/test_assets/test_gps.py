"""
  Test for allantoolkit (https://github.com/aewallin/allantools)
  Stable32 was used to calculate the deviations we compare against.

  GPS tests, AW2016-03-17
"""

import numpy as np
import pathlib
import pytest
import allantoolkit
import allantoolkit.allantools as allan
import allantoolkit.testutils as testutils


# top level directory with asset files
ASSETS_DIR = pathlib.Path(__file__).parent.parent / 'assets'

# input data files, and associated verbosity, tolerance, and acquisition rate
assets = [
    ('gps/gps_1pps_phase_data.txt.gz', 1, 1e-4, 1.),
]

# input result files and function which should replicate them
results = [
    ('stable32_ADEV_decade.txt', allantoolkit.allantools.adev),
    ('stable32_OADEV_octave.txt', allantoolkit.allantools.oadev),
    ('stable32_MDEV_octave.txt', allantoolkit.allantools.mdev),
    ('stable32_TDEV_octave.txt', allantoolkit.allantools.tdev),
    ('stable32_HDEV_octave.txt', allantoolkit.allantools.hdev),
    ('stable32_OHDEV_octave.txt', allantoolkit.allantools.ohdev),
    ('stable32_TOTDEV_octave.txt', allantoolkit.allantools.totdev),
]


@pytest.mark.slow
@pytest.mark.parametrize('datafile, verbose, tolerance, rate', assets)
@pytest.mark.parametrize('result, fct', results)
def test_generic(datafile, result, fct, verbose, tolerance, rate):

    datafile = ASSETS_DIR / datafile
    result = datafile.parent / result

    testutils.test_row_by_row(fct, datafile, rate, result,
                              verbose=verbose, tolerance=tolerance)


# input result files and function which should replicate them
results = [
    ('stable32_ADEV_decade.txt', allantoolkit.allantools.adev),
    ('stable32_OADEV_octave.txt', allantoolkit.allantools.oadev),
    ('stable32_MDEV_octave.txt', allantoolkit.allantools.mdev),
    ('stable32_TDEV_octave.txt', allantoolkit.allantools.tdev),
    ('stable32_HDEV_octave.txt', allantoolkit.allantools.hdev),
    ('stable32_OHDEV_octave.txt', allantoolkit.allantools.ohdev),
]


@pytest.mark.slow
@pytest.mark.parametrize('datafile, verbose, tolerance, rate', assets)
@pytest.mark.parametrize('result, fct', results)
def test_generic_ci_and_noiseID(datafile, result, fct, verbose, tolerance,
                                rate):

    datafile = ASSETS_DIR / datafile
    result = datafile.parent / result

    phase = testutils.read_datafile(datafile)

    s32rows = testutils.read_stable32(resultfile=result, datarate=rate)
    s32taus = [row['tau'] for row in s32rows]

    (taus, devs, errs, ns) = allan.adev(phase, rate=rate, data_type="phase",
                                        taus=s32taus)

    for idx, row in enumerate(s32rows):

        dev = devs[idx]
        try:
            # CI including noise-ID
            (lo2, hi2) = allantoolkit.ci.confidence_interval_noiseID(
                phase, dev, af=int(row['m']), dev_type=str(fct).split('.')[-1],
                data_type="phase")

            assert np.isclose(lo2, row['dev_min'], rtol=1e-2, atol=0)
            assert np.isclose(hi2, row['dev_max'], rtol=1e-2, atol=0)
            print(" CI OK! tau= %f  lo/s32_lo = %f hi/s32_hi = %f " % (
            row['tau'], lo2 / row['dev_min'], hi2 / row['dev_max']))

        except NotImplementedError:
            print("can't do CI for tau= %f" % row['tau'])
            pass


@pytest.mark.slow
@pytest.mark.parametrize('datafile, verbose, tolerance, rate', assets)
def test_noise_id(datafile, verbose, tolerance, rate):
    """ test for noise-identification """

    datafile = ASSETS_DIR / datafile
    result = datafile.parent / 'stable32_ADEV_decade.txt'

    phase = testutils.read_datafile(datafile)
    s32_rows = testutils.read_stable32(result, rate)

    # test noise-ID
    for s32 in s32_rows:
        tau, alpha, AF = s32['tau'], s32['alpha'], int(s32['m'])
        try:
            alpha_int = allantoolkit.ci.autocorr_noise_id(phase, af=AF)[0]
            print(tau, alpha, alpha_int)
            assert alpha_int == alpha
        except NotImplementedError:
            print("can't do noise-ID for tau= %f" % s32['tau'])
