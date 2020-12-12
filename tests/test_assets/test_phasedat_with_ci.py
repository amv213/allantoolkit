"""
  PHASE.DAT test for allantoolkit (https://github.com/aewallin/allantools)
  Stable32 was used to calculate the deviations we compare against.

  PHASE.DAT comes with Stable32 (version 1.53 was used in this case)

  This test is for Confidence Intervals in particular
  
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
    ('phasedat/PHASE.DAT', 1, 1e-4, 1.),
]

# input result files and function which should replicate them
results_ci = [
    ('phase_dat_adev_octave.txt', allantoolkit.allantools.adev, 2,
     allantoolkit.ci.edf_greenhall, False, False),
    ('phase_dat_oadev_octave.txt', allantoolkit.allantools.oadev, 2,
     allantoolkit.ci.edf_greenhall, True, False),
    ('phase_dat_mdev_octave.txt', allantoolkit.allantools.mdev, 2,
     allantoolkit.ci.edf_greenhall, True, True),
    ('phase_dat_tdev_octave.txt', allantoolkit.allantools.tdev, 2,
     allantoolkit.ci.edf_greenhall, True, True),
    ('phase_dat_hdev_octave.txt', allantoolkit.allantools.hdev, 3,
     allantoolkit.ci.edf_greenhall, False, False),
    ('phase_dat_ohdev_octave.txt', allantoolkit.allantools.ohdev, 3,
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

    s32_rows = testutils.read_stable32(result, rate)
    phase = np.array(testutils.read_datafile(datafile))

    (taus, devs, errs, ns) = fct(phase, taus=[s32['tau'] for s32 in s32_rows])

    # separate CI computation
    los = []
    his = []
    for (dev, t, n, s32) in zip(devs, taus, ns, s32_rows):

        # Note FIXED alpha here
        edf2 = ci_fct( alpha=0, d=d, m=t, N=len(phase),
                       overlapping=overlapping, modified=modified)

        (lo, hi) = allantoolkit.ci.confidence_interval(dev=dev, edf=edf2)

        assert np.isclose(lo, s32['dev_min'], rtol=1e-2)
        assert np.isclose(hi, s32['dev_max'], rtol=1e-2)
        print(" alpha=0 FIXED, CI OK! tau = %f" % t)

        los.append(lo)
        his.append(hi)
        try:
            (lo2, hi2) = allantoolkit.ci.confidence_interval_noiseID(
                phase, dev, af=int(t), dev_type=str(fct).split('.')[-1],
                data_type="phase")
            assert np.isclose(lo2, s32['dev_min'], rtol=1e-2)
            assert np.isclose(hi2, s32['dev_max'], rtol=1e-2)
            print(" ACF_NID CI OK! tau = %f" % t)
        except NotImplementedError:
            print("can't do CI for tau = %f" % t)
            pass

    # compare to Stable32
    print("adev()")
    print("    n   tau dev_min  dev      dev_max ")
    for (s32, t2, d2, lo2, hi2, n2) in zip(s32_rows, taus, devs, los, his, ns):
        print("S32 %03d %03.1f %1.6f %1.6f %1.6f" % (
        s32['n'], s32['tau'], s32['dev_min'], s32['dev'], s32['dev_max']))
        print("AT  %03d %03.1f %1.6f %1.6f %1.6f" % (
        n2, t2, round(lo2, 5), round(d2, 5), round(hi2, 5)))
        testutils.check_approx_equal(s32['n'], n2, tolerance=1e-9)
        testutils.check_approx_equal(s32['dev_min'], lo2, tolerance=1e-3)
        testutils.check_approx_equal(s32['dev'], d2, tolerance=1e-4)
        testutils.check_approx_equal(s32['dev_max'], hi2, tolerance=1e-3)
    print("----")



#  Need custom test for totdev due to different edf signature
@pytest.mark.slow
@pytest.mark.parametrize('datafile, verbose, tolerance, rate', assets)
def test_phasedat_totdev(datafile, verbose, tolerance, rate):

    datafile = ASSETS_DIR / datafile
    result = datafile.parent / 'phase_dat_totdev_octave.txt'

    s32_rows = testutils.read_stable32(result, rate)
    phase = testutils.read_datafile(datafile)

    (taus,devs,errs,ns) = allantoolkit.allantools.totdev(
        phase, taus=[s32['tau'] for s32 in s32_rows])

    los=[]
    his=[]
    for (dev, t, n) in zip(devs, taus, ns):

        edf = allantoolkit.ci.edf_totdev(len(phase), t, alpha=0)
        (lo, hi) = allantoolkit.ci.confidence_interval(dev=dev, edf=edf)

        los.append(lo)
        his.append(hi)

    print("totdev()")
    for (s32, t2, d2, lo2, hi2, n2) in zip(s32_rows, taus, devs, los, his, ns):
        print("s32 %03d %03f %1.6f %1.6f %1.6f" % (s32['n'], s32['tau'], s32['dev_min'], s32['dev'], s32['dev_max']))
        print("at  %03d %03f %1.6f %1.6f %1.6f" % (n2, t2, round(lo2,5), round(d2,5), round(hi2,5) ))
        testutils.check_approx_equal(s32['dev_min'], lo2, tolerance=1e-3)
        testutils.check_approx_equal(s32['dev_max'], hi2, tolerance=1e-3)
    print("----")


#  Need custom test for totdev due to different edf signature
# FIXME: failing test that we don't run
@pytest.mark.skip
@pytest.mark.slow
@pytest.mark.parametrize('datafile, verbose, tolerance, rate', assets)
def test_slow_failing_phasedat_mtotdev(datafile, verbose, tolerance, rate):

    datafile = ASSETS_DIR / datafile
    result = datafile.parent / 'phase_dat_mtotdev_octave_alpha0.txt'

    s32_rows = testutils.read_stable32(result, rate)
    phase = testutils.read_datafile(datafile)

    (taus, devs, errs, ns) = allantoolkit.allantools.mtotdev(
        phase, taus=[s32['tau'] for s32 in s32_rows])

    los = []
    his = []
    for (d, t, n) in zip(devs, taus, ns):

        if int(t) < 10:
            edf = allantoolkit.ci.edf_greenhall(
                alpha=0, d=2, m=int(t), N=len(phase),
                overlapping=True, modified=True)
        else:
            edf = allantoolkit.ci.edf_mtotdev(len(phase), t, alpha=0)

        print(edf)
        (lo, hi) = allantoolkit.ci.confidence_interval(dev=d, edf=edf)
        # allan.uncertainty_estimate(len(phase), t, d,ci=0.683,noisetype='wf')
        los.append(lo)
        his.append(hi)

    print("mtotdev()")
    for (s32, t2, d2, lo2, hi2, n2) in zip(s32_rows, taus, devs, los, his,
                                           ns):
        print("s32 %03d %03f %1.6f %1.6f %1.6f" % (
        s32['n'], s32['tau'], s32['dev_min'], s32['dev'], s32['dev_max']))
        print("at  %03d %03f %1.6f %1.6f %1.6f" % (
        n2, t2, round(lo2, 5), round(d2, 5), round(hi2, 5)))
        testutils.check_approx_equal(s32['dev_min'], lo2, tolerance=1e-3)
        testutils.check_approx_equal(s32['dev_max'], hi2, tolerance=1e-3)
    print("----")


@pytest.mark.slow
@pytest.mark.parametrize('datafile, verbose, tolerance, rate', assets)
def test_noise_id(datafile, verbose, tolerance, rate):
    """ test for noise-identification """

    datafile = ASSETS_DIR / datafile
    result = datafile.parent / 'phase_dat_oadev_octave.txt'

    s32_rows = testutils.read_stable32(result, rate)
    phase = testutils.read_datafile(datafile)

    for s32 in s32_rows:
        tau, alpha, af = s32['tau'], s32['alpha'], int(s32['m'])
        try:
            alpha_int = allantoolkit.ci.autocorr_noise_id(phase, af=af)[0]
            assert alpha_int == alpha
            print("OK noise-ID for af = %d" % af)

        except:
            print("can't do noise-ID for af = %d" % af)
        print("tau= %f, alpha= %f, alpha_int = %d" % (tau, alpha, alpha_int))
