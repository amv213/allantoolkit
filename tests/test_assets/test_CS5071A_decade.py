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

data_file = '../assets/Cs5071A/5071A_phase.txt.gz'  # input data file
data_file = pathlib.Path(data_file).resolve()

verbose = 1
tolerance = 1e-4
rate = 1.  # stable32 runs were done with this data-interval


def generic_test(datafile: pathlib.Path = data_file, result="", fct=None):
    result = datafile.parent / result
    testutils.test_row_by_row(fct, datafile, 1.0, result,
                              verbose=verbose, tolerance=tolerance)


@pytest.mark.slow
def test_adev():
    generic_test(result='adev_decade.txt', fct=allantoolkit.allantools.adev)

@pytest.mark.slow
def test_adev_ci():
    s32rows = testutils.read_stable32(resultfile='adev_decade.txt', datarate=1.0)
    for row in s32rows:
        data = testutils.read_datafile(data_file)
        (taus, devs, errs, ns) = allantoolkit.allantools.adev(
            data, rate=rate, taus=[row['tau']])
        edf = allantoolkit.ci.edf_greenhall(
            alpha=row['alpha'], d=2, m=row['m'], N=len(data),
            overlapping=False, modified = False, verbose=True)
        (lo, hi) = allantoolkit.ci.confidence_interval(
            devs[0], ci=0.68268949213708585, edf=edf)

        print("n check: ", testutils.check_equal( ns[0], row['n']))
        print("dev check: ", testutils.check_approx_equal( devs[0], row['dev'] ) )
        print("min dev check: ",  lo, row['dev_min'], testutils.check_approx_equal( lo, row['dev_min'], tolerance=1e-3 ) )
        print("max dev check: ", hi, row['dev_max'], testutils.check_approx_equal( hi, row['dev_max'], tolerance=1e-3 ) )


@pytest.mark.slow
def test_oadev():
    generic_test(result='oadev_decade.txt', fct=allantoolkit.allantools.oadev)


@pytest.mark.slow
def test_oadev_ci():
    s32rows = testutils.read_stable32(resultfile='oadev_decade.txt',
                                      datarate=1.0)
    for row in s32rows:
        data = testutils.read_datafile(data_file)
        (taus, devs, errs, ns) = allantoolkit.allantools.oadev(
            data, rate=rate, taus=[ row['tau']])
        edf = allantoolkit.ci.edf_greenhall(
            alpha=row['alpha'], d=2, m=row['m'], N=len(data),
            overlapping=True, modified = False, verbose=True)

        (lo, hi) = allantoolkit.ci.confidence_interval(
            devs[0], ci=0.68268949213708585, edf=edf)
        print("n check: ", testutils.check_equal( ns[0], row['n'] ) )
        print("dev check: ", testutils.check_approx_equal( devs[0], row['dev'] ) )
        print("min dev check: ",  lo, row['dev_min'], testutils.check_approx_equal( lo, row['dev_min'], tolerance=1e-3 ) )
        print("max dev check: ", hi, row['dev_max'], testutils.check_approx_equal( hi, row['dev_max'], tolerance=1e-3 ) )


@pytest.mark.slow
def test_mdev():
    generic_test(result='mdev_decade.txt', fct=allantoolkit.allantools.mdev)


@pytest.mark.slow
def test_mdev_ci():
    s32rows = testutils.read_stable32(resultfile='mdev_decade.txt',
                                      datarate=1.0)
    for row in s32rows:
        data = testutils.read_datafile(data_file)
        (taus, devs, errs, ns) = allantoolkit.allantools.mdev(
            data, rate=rate, taus=[row['tau']])
        edf = allantoolkit.ci.edf_greenhall(
            alpha=row['alpha'], d=2, m=row['m'],  N=len(data),
            overlapping=True, modified = True, verbose=True)
        (lo, hi) =allantoolkit.ci.confidence_interval(
            devs[0], ci=0.68268949213708585, edf=edf)
        print("n check: ", testutils.check_equal( ns[0], row['n'] ) )
        print("dev check: ", testutils.check_approx_equal( devs[0], row['dev'] ) )
        print("min dev check: ",  lo, row['dev_min'], testutils.check_approx_equal( lo, row['dev_min'], tolerance=1e-3 ) )
        print("max dev check: ", hi, row['dev_max'], testutils.check_approx_equal( hi, row['dev_max'], tolerance=1e-3 ) )


@pytest.mark.slow
def test_tdev():
    generic_test(result='tdev_decade.txt', fct=allantoolkit.allantools.tdev)


@pytest.mark.slow
def test_hdev():
    generic_test(result='hdev_decade.txt', fct= allantoolkit.allantools.hdev)


@pytest.mark.slow
def test_hdev_ci():
    s32rows = testutils.read_stable32(resultfile='hdev_decade.txt',
                                      datarate=1.0)
    for row in s32rows:
        data = testutils.read_datafile(data_file)
        (taus, devs, errs, ns) = allantoolkit.allantools.hdev(
            data, rate=rate, taus=[row['tau']])
        edf = allantoolkit.ci.edf_greenhall(
            alpha=row['alpha'], d=3, m=row['m'], N=len(data),
            overlapping=False, modified=False, verbose=True)

        (lo, hi) =allantoolkit.ci.confidence_interval(
            devs[0], ci=0.68268949213708585, edf=edf)

        print("n check: ", testutils.check_equal( ns[0], row['n'] ) )
        print("dev check: ", testutils.check_approx_equal( devs[0], row['dev'] ) )
        print("min dev check: ",  lo, row['dev_min'], testutils.check_approx_equal( lo, row['dev_min'], tolerance=1e-3 ) )
        print("max dev check: ", hi, row['dev_max'], testutils.check_approx_equal( hi, row['dev_max'], tolerance=1e-3 ) )


@pytest.mark.slow
def test_ohdev():
    generic_test(result='ohdev_decade.txt', fct=allantoolkit.allantools.ohdev)


@pytest.mark.slow
def test_ohdev_ci():
    s32rows = testutils.read_stable32(resultfile='ohdev_decade.txt',
                                      datarate=1.0)
    for row in s32rows:
        data = testutils.read_datafile(data_file)
        (taus, devs, errs, ns) = allantoolkit.allantools.ohdev(
            data, rate=rate, taus=[row['tau']])
        edf = allantoolkit.ci.edf_greenhall(
            alpha=row['alpha'], d=3, m=row['m'], N=len(data),
            overlapping=True, modified = False, verbose=True)

        (lo,hi) =allantoolkit.ci.confidence_interval(
            devs[0], ci=0.68268949213708585, edf=edf)

        print("n check: ", testutils.check_equal( ns[0], row['n'] ) )
        print("dev check: ", testutils.check_approx_equal( devs[0], row['dev'] ) )
        print("min dev check: ",  lo, row['dev_min'], testutils.check_approx_equal( lo, row['dev_min'], tolerance=1e-3 ) )
        print("max dev check: ", hi, row['dev_max'], testutils.check_approx_equal( hi, row['dev_max'], tolerance=1e-3 ) )


@pytest.mark.slow
def test_totdev():
    generic_test(result='totdev_decade.txt',
                 fct=allantoolkit.allantools.totdev)


@pytest.mark.slow
def test_totdev_ci():
    s32rows = testutils.read_stable32(resultfile='totdev_decade.txt',
                                      datarate=1.0)
    for row in s32rows:
        data = testutils.read_datafile(data_file)
        (taus, devs, errs, ns) = allantoolkit.allantools.totdev(
            data, rate=rate, taus=[ row['tau']])
        edf = allantoolkit.ci.edf_totdev(
            N=len(data), m=row['m'], alpha=row['alpha'])

        #,d=3,m=row['m'],N=len(data),overlapping=True, modified = False, verbose=True)
        (lo,hi) = allantoolkit.ci.confidence_interval(
            devs[0], ci=0.68268949213708585, edf=edf)

        print("n check: ", testutils.check_equal( ns[0], row['n'] ) )
        print("dev check: ", testutils.check_approx_equal( devs[0], row['dev'] ) )
        print("min dev check: ",  lo, row['dev_min'], testutils.check_approx_equal( lo, row['dev_min'], tolerance=1e-3 ) )
        print("max dev check: ", hi, row['dev_max'], testutils.check_approx_equal( hi, row['dev_max'], tolerance=1e-3 ) )


# def test_mtie():
#    generic_test( result='mtie_fast.txt' , fct= allan.mtie )

@pytest.mark.slow
def test_tierms():
    generic_test(result='tierms_decade.txt',
                 fct=allantoolkit.allantools.tierms)
