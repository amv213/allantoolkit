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

import os
import time
import pytest
import allantoolkit
import allantoolkit.testutils as testutils


def print_elapsed(start):
    end = time.clock()
    print(" %.2f s"% ( end-start ))
    return time.clock()


def change_to_test_dir():
    # hack to run script from its own directory
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)


data_file = '5071A_phase.txt.gz'  # input data file
verbose = 1
tolerance = 1e-4
rate = 1/float(1.0)  # stable32 runs were done with this data-interval


@pytest.mark.slow
class TestCS:
    def test_adev(self):
        self.generic_test(result='adev_decade.txt',
                          fct=allantoolkit.allantools.adev)

    def test_adev_ci(self):
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

    def test_oadev(self):
        self.generic_test(result='oadev_decade.txt',
                          fct=allantoolkit.allantools.oadev)

    def test_oadev_ci(self):
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

    def test_mdev(self):
        self.generic_test(result='mdev_decade.txt',
                          fct=allantoolkit.allantools.mdev)

    def test_mdev_ci(self):
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

    def test_tdev(self):
        self.generic_test(result='tdev_decade.txt',
                          fct=allantoolkit.allantools.tdev)

    def test_hdev(self):
        self.generic_test(result='hdev_decade.txt',
                          fct= allantoolkit.allantools.hdev)

    def test_hdev_ci(self):
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

    def test_ohdev(self):
        self.generic_test(result='ohdev_decade.txt',
                          fct=allantoolkit.allantools.ohdev)

    def test_ohdev_ci(self):
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

    def test_totdev(self):
        self.generic_test(result='totdev_decade.txt',
                          fct=allantoolkit.allantools.totdev)

    def test_totdev_ci(self):
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

    # def test_mtie(self):
    #    self.generic_test( result='mtie_fast.txt' , fct= allan.mtie )

    def test_tierms(self):
        self.generic_test(result='tierms_decade.txt',
                          fct=allantoolkit.allantools.tierms)

    @staticmethod
    def generic_test(datafile=data_file, result="", fct=None):
        change_to_test_dir()
        testutils.test_row_by_row(fct, datafile, 1.0, result,
                                  verbose=verbose, tolerance=tolerance)
