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
import pathlib
import allantoolkit
import allantoolkit.testutils as testutils

# top level directory with asset files
ASSETS_DIR = pathlib.Path(__file__).parent.parent / 'assets'

# input data files, and associated verbosity, tolerance, and acquisition rate
assets = [
    ('Cs5071A/5071A_phase.txt.gz', 1, 1e-4, 1.),
]

# input result files and function which should replicate them
results = [
    ('adev_decade.txt', allantoolkit.allantools.adev),
    ('oadev_decade.txt', allantoolkit.allantools.oadev),
    ('mdev_decade.txt', allantoolkit.allantools.mdev),
    ('tdev_decade.txt', allantoolkit.allantools.tdev),
    ('hdev_decade.txt', allantoolkit.allantools.hdev),
    ('ohdev_decade.txt', allantoolkit.allantools.ohdev),
    ('totdev_decade.txt', allantoolkit.allantools.totdev),
    ('tierms_decade.txt', allantoolkit.allantools.tierms),
]


@pytest.mark.slow
@pytest.mark.parametrize('datafile, verbose, tolerance, rate', assets)
@pytest.mark.parametrize('result, fct', results)
def test_generic(datafile, result, fct, verbose, tolerance, rate):

    datafile = ASSETS_DIR / datafile
    result = datafile.parent / result

    testutils.test_row_by_row(fct, datafile, rate, result,
                              verbose=verbose, tolerance=tolerance)


results_ci = [
    ('adev_decade.txt', allantoolkit.allantools.adev, 2,
     allantoolkit.ci.edf_greenhall, False, False),
    ('oadev_decade.txt', allantoolkit.allantools.oadev, 2,
     allantoolkit.ci.edf_greenhall, True, False),
    ('mdev_decade.txt', allantoolkit.allantools.mdev, 2,
     allantoolkit.ci.edf_greenhall, True, True),
    ('hdev_decade.txt', allantoolkit.allantools.hdev, 3,
     allantoolkit.ci.edf_greenhall, False, False),
    ('ohdev_decade.txt', allantoolkit.allantools.ohdev, 3,
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

        (taus, devs, errs, ns) = fct(data, rate=rate, taus=[row['tau']])

        edf = ci_fct(
            alpha=row['alpha'], d=d, m=row['m'], N=len(data),
            overlapping=overlapping, modified=modified, verbose=True)

        (lo, hi) = allantoolkit.ci.confidence_interval(
            devs[0], ci=0.68268949213708585, edf=edf)

        print("n check: ", testutils.check_equal( ns[0], row['n']))
        print("dev check: ", testutils.check_approx_equal( devs[0], row['dev'] ) )
        print("min dev check: ",  lo, row['dev_min'], testutils.check_approx_equal( lo, row['dev_min'], tolerance=1e-3 ) )
        print("max dev check: ", hi, row['dev_max'], testutils.check_approx_equal( hi, row['dev_max'], tolerance=1e-3 ) )


#  Need custom test for totdev due to different edf signature
@pytest.mark.slow
@pytest.mark.parametrize('datafile, verbose, tolerance, rate', assets)
def test_totdev_ci(datafile, verbose, tolerance, rate):

    datafile = ASSETS_DIR / datafile
    result = datafile.parent / 'totdev_decade.txt'

    s32rows = testutils.read_stable32(resultfile=result, datarate=rate)

    for row in s32rows:

        data = testutils.read_datafile(datafile)

        (taus, devs, errs, ns) = allantoolkit.allantools.totdev(
            data, rate=rate, taus=[row['tau']])

        edf = allantoolkit.ci.edf_totdev(
            alpha=row['alpha'], m=row['m'], N=len(data))

        (lo, hi) = allantoolkit.ci.confidence_interval(
            devs[0], ci=0.68268949213708585, edf=edf)

        print("n check: ", testutils.check_equal( ns[0], row['n']))
        print("dev check: ", testutils.check_approx_equal( devs[0], row['dev'] ) )
        print("min dev check: ",  lo, row['dev_min'], testutils.check_approx_equal( lo, row['dev_min'], tolerance=1e-3 ) )
        print("max dev check: ", hi, row['dev_max'], testutils.check_approx_equal( hi, row['dev_max'], tolerance=1e-3 ) )
