"""Test data based on a Strontium Optical Lattice Clock. Reference results
were generated with Stable32 on the data in plot3.txt

Data type: fractional frequency
Sampling rate: 0.4 Hz
Gaps: No

This suite of test is useful to check that Allantoolkit works in the same
way as Stable32 also on data acquired as fractional frequency. This suite of
test will not check if gaps are handled in the same way, as the dataset has
no gaps.
"""

import allantoolkit
import pathlib
import pytest
import numpy as np

# top level directory with original (frequency) data for these tests
ASSETS_DIR = pathlib.Path(__file__).parent.parent / 'assets/Sr87'

# Raw frequency data collected from experiment
Y = allantoolkit.testutils.read_datafile(ASSETS_DIR / 'freq/freq_data.txt')

# Phase and normalised phase data, obtained by Stable32 conversion of freq
X = allantoolkit.testutils.read_datafile(ASSETS_DIR / 'phase/phase_data.txt')
X0 = allantoolkit.testutils.read_datafile(ASSETS_DIR /
                                          'phase0/phase0_data.txt')

# Data sampling rate
RATE = 0.4  # Hz, tau_0 = 2.5s


# INPUT DATA MANIPULATIONS:

def test_frequency2phase():
    """Test that fractional frequency data is converted to phase data in the
    same way as Stable32. This will make sure all Stable32 deviations
    calculated on frequency data should match allantoolkit deviations which
    are (mostly) calculated on phase data behind the scenes."""
    output = allantoolkit.utils.frequency2phase(y=Y, rate=RATE,
                                                normalize=False)

    assert np.allclose(X, output, atol=1e-32)


def test_frequency2phase0():
    """Test that fractional frequency data is converted to phase data in the
    same way as Stable32. This will make sure all Stable32 deviations
    calculated on frequency data should match allantoolkit deviations which
    are (mostly) calculated on phase data behind the scenes."""
    output = allantoolkit.utils.frequency2phase(y=Y, rate=RATE,
                                                normalize=True)

    assert np.allclose(X0, output, atol=1e-28)
    # Note: Looks like Stable32 normalisation introduces discrepancies at
    # the 1e-29 level


def test_phase2frequency():
    """Test that can losslessly convert fractional frequency data to phase
    data, and back."""

    x = allantoolkit.utils.frequency2phase(y=Y, rate=RATE, normalize=False)

    y = allantoolkit.utils.phase2frequency(x=x, rate=RATE)

    assert np.allclose(Y, y, atol=1e-32)


def test_phase02frequency():
    """Test that can losslessly convert fractional frequency data to
    normalised phase data, and back."""

    x0 = allantoolkit.utils.frequency2phase(y=Y, rate=RATE, normalize=True)

    y = allantoolkit.utils.phase2frequency(x=x0, rate=RATE)

    assert np.allclose(Y, y, atol=1e-32)


# TEST RESULTS

input_data = [
    (X0, 'phase'),
    (Y, 'freq'),
]

tau_types = [
    'octave',
    'decade',
]

fcts = [
    allantoolkit.allantools.adev,
    #allantoolkit.allantools.oadev,
    #allantoolkit.allantools.mdev,
    #allantoolkit.allantools.tdev,
    #allantoolkit.allantools.hdev,
    #allantoolkit.allantools.ohdev,
    #allantoolkit.allantools.totdev,
    #pytest.param(allantoolkit.allantools.mtotdev,  marks=pytest.mark.slow),
    #pytest.param(allantoolkit.allantools.ttotdev, marks=pytest.mark.slow),
    #pytest.param(allantoolkit.allantools.htotdev, marks=pytest.mark.slow),
    #allantoolkit.allantools.theo1
]


@pytest.mark.parametrize('data, data_type', input_data)
@pytest.mark.parametrize('taus', tau_types)
@pytest.mark.parametrize('func', fcts)
def test_dev(data, data_type, func, taus):

    if data_type == 'freq':
        fn = ASSETS_DIR / data_type / taus / (func.__name__ + '.txt')
    else:
        fn = ASSETS_DIR / (data_type + '0') / taus / (func.__name__ + '.txt')

    expected = allantoolkit.testutils.read_stable32(fn)

    output = func(data=data, rate=RATE, data_type=data_type, taus=taus)

    afs2, taus2, ns2, alphas2, devs2, errs_lo2, errs_hi2 = output
    devminus2, devplus2 = devs2 - errs_lo2, devs2 + errs_hi2
    devs2 = [float(np.format_float_scientific(dev, 4)) for dev in devs2]
    devminus2 = [float(np.format_float_scientific(lo, 4)) for lo in devminus2]
    devplus2 = [float(np.format_float_scientific(hi, 4)) for hi in devplus2]

    for i, row in enumerate(expected):

        af, tau, n, alpha, minus, dev, plus = row.T
        af, n, alpha = int(af), int(n), int(alpha)

        af2, tau2, n2, alpha2, dev2, minus2, plus2 = \
            afs2[i], taus2[i], ns2[i], alphas2[i], devs2[i], devminus2[i], \
            devplus2[i]

        print("AF  TAU   #   ALPHA   DEV_LO  DEV   DEV_HI")
        print([af, tau, n, alpha, minus, dev, plus], '<-REF')
        print([af2, tau2, n2, alpha2, minus2, dev2, plus2], '<-ME')

        #assert af == af2, f'S32:{af} vs. AT {af2}'
        #assert tau == tau2, f'S32:{tau} vs. AT {tau2}'
        #assert n == n2, f'S32:{n} vs. AT {n2}'
        #assert alpha == alpha2, f'S32:{alpha} vs. AT {alpha2}'
        #assert minus == minus2, f'S32:\n{minus}\nvs.\nAT:\n{minus2}'
        #assert dev == dev2, f'S32:\n{dev}\nvs.\nAT:\n{dev2}'
        #assert plus == plus2, f'S32:\n{plus}\nvs.\nAT:\n{plus2}'

    assert 1 == 2

'''

tau_types = [
    'octave',
    'decade',
]

fcts = [
    allantoolkit.allantools.mtie,
    allantoolkit.allantools.tierms
]


@pytest.mark.parametrize('taus', tau_types)
@pytest.mark.parametrize('func', fcts)
def test_xtras(func, taus):

    fn = ASSETS_DIR / 'phase0' / taus / (func.__name__ + '.txt')

    expected = allantoolkit.testutils.read_stable32(fn, file_type='tie')

    output = func(data=X0, rate=RATE, data_type='phase', taus=taus)

    afs2, taus2, ns2, alphas2, devs2, errs_lo2, errs_hi2 = output
    devs2 = [float(np.format_float_scientific(dev, 4)) for dev in devs2]

    for i, row in enumerate(expected):

        af, tau, n, dev = row.T
        af, n = int(af), int(n)

        af2, tau2, n2, alpha2, dev2, err_lo2, err_hi2 = \
            afs2[i], taus2[i], ns2[i], alphas2[i], devs2[i], errs_lo2[i], \
            errs_hi2[i]

        print("AF  TAU   #    DEV")
        print([af, tau, n, dev], '<-REF')
        print([af2, tau2, n2, dev2], '<-ME')

        assert af == af2, f'S32:{af} vs. AT {af2}'
        assert tau == tau2, f'S32:{tau} vs. AT {tau2}'
        assert n == n2, f'S32:{n} vs. AT {n2}'
        assert dev == dev2, f'S32:\n{dev}\nvs.\nAT:\n{dev2}'

'''


'''
# input result files and function which should replicate them
fcts = [
    allantoolkit.allantools.adev,
    allantoolkit.allantools.oadev,
    allantoolkit.allantools.mdev,
    allantoolkit.allantools.tdev,
    allantoolkit.allantools.hdev,
    allantoolkit.allantools.ohdev,
    allantoolkit.allantools.totdev,
    pytest.param(allantoolkit.allantools.mtotdev, marks=pytest.mark.slow),
    pytest.param(allantoolkit.allantools.ttotdev, marks=pytest.mark.slow),
    pytest.param(allantoolkit.allantools.htotdev, marks=pytest.mark.slow),
]


@pytest.mark.parametrize('fct', fcts)
def test_noise_id_phase0_octave(fct):
    """ test for noise-identification """

    datafile = ASSETS_DIR / 'phase0/phase0_data.txt'

    result_fn = fct.__name__ + '.txt'
    resultfile = ASSETS_DIR / 'phase0/octave' / result_fn

    phase = allantoolkit.testutils.read_datafile(datafile)
    s32_rows = allantoolkit.testutils.read_stable32(resultfile)

    # test noise-ID
    for s32 in s32_rows[:-1, :]:  # only test up to one to-last

        m, tau, n, alpha = int(s32[0]), s32[1], int(s32[2]), int(s32[3])

        alpha_int = allantoolkit.ci.noise_id(phase, data_type='phase',
                                             m=m, tau=tau,
                                             dev_type=fct.__name__,
                                             n=n)

        if alpha_int != -99:  # not implemented token

            print(fct.__name__, tau, alpha, alpha_int)

            assert alpha_int == alpha


@pytest.mark.parametrize('fct', fcts)
def test_noise_id_phase0_all(fct):
    """ test for noise-identification """

    datafile = ASSETS_DIR / 'phase0/phase0_data.txt'

    result_fn = fct.__name__ + '.txt'
    resultfile = ASSETS_DIR / 'phase0/all' / result_fn

    phase = allantoolkit.testutils.read_datafile(datafile)
    s32_rows = allantoolkit.testutils.read_stable32(resultfile)

    # test noise-ID
    for s32 in s32_rows:

        m, tau, n, alpha = int(s32[0]), s32[1], int(s32[2]), int(s32[3])

        alpha_int = allantoolkit.ci.noise_id(phase, data_type='phase',
                                             m=m, tau=tau,
                                             dev_type=fct.__name__,
                                             n=n)

        if alpha_int != -99:  # not implemented token

            print(fct.__name__, tau, alpha, alpha_int)

            assert alpha_int == alpha




@pytest.mark.parametrize('fct', fcts)
def test_noise_id_freq_octave(fct):
    """ test for noise-identification """

    datafile = ASSETS_DIR / 'freq/freq_data.txt'

    result_fn = fct.__name__ + '.txt'
    resultfile = ASSETS_DIR / 'freq/octave' / result_fn

    phase = allantoolkit.testutils.read_datafile(datafile)
    s32_rows = allantoolkit.testutils.read_stable32(resultfile)

    # test noise-ID
    for s32 in s32_rows:
        m, tau, n, alpha = int(s32[0]), s32[1], int(s32[2]), int(s32[3])

        alpha_int = allantoolkit.ci.noise_id(phase, data_type='freq',
                                             m=m, tau=tau,
                                             dev_type=fct.__name__,
                                             n=n)

        if alpha_int != -99:  # not implemented token
            print(f"{fct.__name__} @ tau {tau} should have alpha {alpha} and "
                  f"not {alpha_int}")

            assert alpha_int == alpha


@pytest.mark.parametrize('fct', fcts)
def test_noise_id_freq_decade(fct):
    """ test for noise-identification """

    datafile = ASSETS_DIR / 'freq/freq_data.txt'

    result_fn = fct.__name__ + '.txt'
    resultfile = ASSETS_DIR / 'freq/decade' / result_fn

    phase = allantoolkit.testutils.read_datafile(datafile)
    s32_rows = allantoolkit.testutils.read_stable32(resultfile)

    # test noise-ID
    for s32 in s32_rows:
        m, tau, n, alpha = int(s32[0]), s32[1], int(s32[2]), int(s32[3])


        alpha_int = allantoolkit.ci.noise_id(phase, data_type='freq',
                                             m=m, tau=tau,
                                             dev_type=fct.__name__,
                                             n=n)

        if alpha_int != -99:  # not implemented token
            print(f"{fct.__name__} @ tau {tau} should have alpha {alpha}:"
                  f"you have {alpha_int}")

            assert alpha_int == alpha



# input result files and function which should replicate them
fcts = [
    allantoolkit.allantools.adev,
    allantoolkit.allantools.oadev,
    allantoolkit.allantools.mdev,
    allantoolkit.allantools.tdev,
    allantoolkit.allantools.hdev,
    allantoolkit.allantools.ohdev,
    allantoolkit.allantools.totdev,
    pytest.param(allantoolkit.allantools.mtotdev, marks=pytest.mark.slow),
    pytest.param(allantoolkit.allantools.ttotdev, marks=pytest.mark.slow),
    pytest.param(allantoolkit.allantools.htotdev, marks=pytest.mark.slow),
    allantoolkit.allantools.theo1,
    pytest.param(allantoolkit.allantools.mtie, marks=pytest.mark.slow),
    allantoolkit.allantools.tierms,
]

# FIXME: add tests of noise type identified, and upper and minimum error
#  bounds once implemented those calculations in allantoolkit
@pytest.mark.parametrize('fct', fcts)
def test_generic_phase_octave(fct):
    """Test allantoolkit deviations calculated on phase data at octave
    averaging times, give the same results as Stable32 deviations calculated
    on phase data at octave averaging times"""

    datafile = ASSETS_DIR / 'phase0/phase0_data.txt'

    result_fn = fct.__name__ + '.txt'
    resultfile = ASSETS_DIR / 'phase0/octave' / result_fn

    return allantoolkit.testutils.test_row_by_row(fct, datafile, RATE,
                                                  resultfile, tolerance=1e-4)


@pytest.mark.parametrize('fct', fcts)
def test_generic_phase_decade(fct):
    """Test allantoolkit deviations calculated on phase data at octave
    averaging times, give the same results as Stable32 deviations calculated
    on phase data at octave averaging times"""

    datafile = ASSETS_DIR / 'phase0/phase0_data.txt'

    result_fn = fct.__name__ + '.txt'
    resultfile = ASSETS_DIR / 'phase0/decade' / result_fn

    return allantoolkit.testutils.test_row_by_row(fct, datafile, RATE,
                                                  resultfile, tolerance=1e-4)


@pytest.mark.parametrize('fct', fcts)
def test_generic_phase_all(fct):
    """Test allantoolkit deviations calculated on phase data at octave
    averaging times, give the same results as Stable32 deviations calculated
    on phase data at octave averaging times"""

    datafile = ASSETS_DIR / 'phase0/phase0_data.txt'

    result_fn = fct.__name__ + '.txt'
    resultfile = ASSETS_DIR / 'phase0/all' / result_fn

    return allantoolkit.testutils.test_row_by_row(fct, datafile, RATE,
                                                  resultfile, tolerance=1e-4)
'''