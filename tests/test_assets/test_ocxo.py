"""
  Test for allantoolkit (https://github.com/aewallin/allantools)
  Stable32 was used to calculate the deviations we compare against.

  AW2015-06-26
  
  The dataset is from the 10 MHz output at the back of an HP Impedance Analyzer
  measured with Keysight 53230A counter, 1.0s gate, RCON mode, with H-maser 10MHz reference

"""

import pathlib
import pytest
import allantoolkit
import allantoolkit.testutils as testutils


# top level directory with asset files
ASSETS_DIR = pathlib.Path(__file__).parent.parent / 'assets'

# input data files, and associated verbosity, tolerance, and acquisition rate
assets = [
    ('ocxo/ocxo_frequency.txt', 1, 1e-4, 1.),
]


# input result files and function which should replicate them
results_ci = [
    ('adev_octave.txt', allantoolkit.allantools.adev, 2,
     allantoolkit.ci.edf_greenhall, False, False),
    ('oadev_octave.txt', allantoolkit.allantools.oadev, 2,
     allantoolkit.ci.edf_greenhall, True, False),
    ('mdev_octave.txt', allantoolkit.allantools.mdev, 2,
     allantoolkit.ci.edf_greenhall, True, True),
    ('tdev_octave.txt', allantoolkit.allantools.tdev, 2,
     allantoolkit.ci.edf_greenhall, True, True),
    ('hdev_octave.txt', allantoolkit.allantools.hdev, 3,
     allantoolkit.ci.edf_greenhall, False, False),
    ('ohdev_octave.txt', allantoolkit.allantools.ohdev, 3,
     allantoolkit.ci.edf_greenhall, True, False),
]


@pytest.mark.slow
@pytest.mark.parametrize('datafile, verbose, tolerance, rate', assets)
@pytest.mark.parametrize('result, fct, d, ci_fct, overlapping, modified',
                         results_ci)
def test_generic_ci(datafile, result, fct, verbose, tolerance, rate,
                    d, ci_fct, overlapping, modified):

    datafile = ASSETS_DIR / datafile
    result = datafile.parent / result

    s32rows = testutils.read_stable32(resultfile=result, datarate=rate)

    for row in s32rows:
        data = testutils.read_datafile(datafile)
        data = allantoolkit.allantools.frequency2fractional(
            data, mean_frequency=1.0e7)
        (taus, devs, errs, ns) = fct(data, rate=rate, data_type="freq",
                                     taus=[row['tau']])

        # NOTE! Here we use alhpa from Stable32-results for the allantools edf computation!
        edf = ci_fct(alpha=row['alpha'], d=d, m=row['m'], N=len(data),
                     overlapping=overlapping, modified=modified, verbose=True)

        (lo, hi) = allantoolkit.ci.confidence_interval(devs[0], edf=edf)

        print("n check: ", testutils.check_equal(ns[0], row['n']))
        print("dev check: ", devs[0], row['dev'],
              testutils.check_approx_equal(devs[0], row['dev'],
                                           tolerance=2e-3))
        print("min dev check: ", lo, row['dev_min'],
              testutils.check_approx_equal(lo, row['dev_min'], tolerance=2e-3))
        print("max dev check: ", hi, row['dev_max'],
              testutils.check_approx_equal(hi, row['dev_max'], tolerance=5e-3))


#  Need custom test for totdev due to different edf signature
# fails, totdev() needs bias-correction, depending on alpha(?)
@pytest.mark.skip(reason="needs bias-correction and noise-ID to work")
@pytest.mark.xfail
@pytest.mark.parametrize('datafile, verbose, tolerance, rate', assets)
def test_totdev_ci(datafile, verbose, tolerance, rate):

    datafile = ASSETS_DIR / datafile
    result = datafile.parent / 'totdev_octave.txt'

    s32rows = testutils.read_stable32(resultfile=result, datarate=rate)

    for row in s32rows:
        data = testutils.read_datafile(datafile)
        data = allantoolkit.allantools.frequency2fractional(
            data, mean_frequency=1.0e7)
        (taus, devs, errs, ns) = allantoolkit.allantools.totdev(
            data, rate=rate, data_type="freq", taus=[row['tau']])
        edf = allantoolkit.ci.edf_totdev(
            N=len(data), m=row['m'], alpha=row['alpha'])

        (lo, hi) = allantoolkit.ci.confidence_interval(devs[0], edf=edf)

        print("n check: ", testutils.check_equal(ns[0], row['n']))
        print("dev check: ", testutils.check_approx_equal(devs[0], row['dev'],
                                                          tolerance=2e-3))
        print("min dev check: %.4g %.4g %d" % (
            lo, row['dev_min'], testutils.check_approx_equal(
                lo, row['dev_min'], tolerance=2e-3)))
        print("max dev check: %.4g %.4g %d" % (
            hi, row['dev_max'], testutils.check_approx_equal(
                hi, row['dev_max'], tolerance=2e-3)))


@pytest.mark.slow
@pytest.mark.parametrize('datafile, verbose, tolerance, rate', assets)
def test_noise_id(datafile, verbose, tolerance, rate):
    """ test for noise-identification """

    datafile = ASSETS_DIR / datafile
    result = datafile.parent / 'mdev_octave.txt'

    s32_rows = testutils.read_stable32(result, rate)

    freq = testutils.read_datafile(datafile)
    phase = allantoolkit.allantools.frequency2phase(freq, rate)

    for s32 in s32_rows:
        s32_tau, s32_alpha, s32_AF = s32['tau'], s32['alpha'], int(
            s32['m'])

        # noise-ID from frequency
        if len(phase) / s32_AF > 20:
            alpha_int, alpha, d, rho = allantoolkit.ci.autocorr_noise_id(
                freq, data_type='freq', af=s32_AF)

            print("y: ", s32_tau, s32_alpha, alpha_int, alpha, rho, d)
            assert alpha_int == s32_alpha

            # noise-ID from phase
        if len(phase) / s32_AF > 20:
            alpha_int, alpha, d, rho = allantoolkit.ci.autocorr_noise_id(
                phase, data_type='phase', af=s32_AF)
            print("x: ", s32_tau, s32_alpha, alpha_int, alpha, rho, d)
            assert alpha_int == s32_alpha
