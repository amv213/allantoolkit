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


# FIXME: update this to use the latest noise_id algorithm
# FIXME: this always fails because dev_type is passed wrong (should be
#  fct.__name__)
@pytest.mark.slow
@pytest.mark.parametrize('datafile, verbose, tolerance, rate', assets)
@pytest.mark.parametrize('result, fct', results)
def test_generic_ci_and_noiseID(datafile, result, fct, verbose, tolerance,
                                rate):

    datafile = ASSETS_DIR / datafile
    result = datafile.parent / result

    phase = testutils.read_datafile(datafile)

    s32rows = testutils.read_stable32(result)
    s32taus = np.array([row[1] for row in s32rows])

    (taus, devs, errs_lo, errs_hi, ns) = allan.adev(phase, rate=rate,
                                                    data_type="phase",
                                                    taus=s32taus)

    for idx, row in enumerate(s32rows):

        dev = devs[idx]
        try:
            # CI including noise-ID
            (lo2, hi2) = allantoolkit.ci.confidence_interval_noiseID(
                phase, dev, af=int(row[0]), dev_type=str(fct).split('.')[-1],
                data_type="phase")

            assert np.isclose(lo2, row[4], rtol=1e-2, atol=0)
            assert np.isclose(hi2, row[6], rtol=1e-2, atol=0)
            print(" CI OK! tau= %f  alpha=%i lo/s32_lo = %f hi/s32_hi = %f "
                  % (row[1], row[3], lo2 / row[4], hi2 / row[6]))

        except NotImplementedError:
            print("can't do CI for tau= %f" % row[1])
            pass


@pytest.mark.slow
@pytest.mark.parametrize('datafile, verbose, tolerance, rate', assets)
@pytest.mark.parametrize('result, fct', results)
def test_noise_id(datafile, result, fct, verbose, tolerance, rate):
    """ test for noise-identification """

    datafile = ASSETS_DIR / datafile
    result = datafile.parent / result

    phase = testutils.read_datafile(datafile)
    s32_rows = testutils.read_stable32(result)

    # test noise-ID
    for s32 in s32_rows:

        m, tau, n, alpha = int(s32[0]), s32[1], int(s32[2]), int(s32[3])

        alpha_int = allantoolkit.ci.noise_id(phase, data_type='phase',
                                             m=m, dev_type=fct.__name__,
                                             n=n)

        if alpha_int != -99:  # not implemented token

            print(fct.__name__, tau, alpha, alpha_int)

            assert alpha_int == alpha
