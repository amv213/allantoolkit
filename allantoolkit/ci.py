"""
this file is part of allantoolkit, https://github.com/aewallin/allantools

- functions for confidence intervals
- functions for noise identification
- functions for computing equivalent degrees of freedom
"""

import logging
import numpy as np
import scipy.special
import scipy.stats  # used in confidence_intervals()
import scipy.signal  # decimation in lag-1 acf
from . import allantools
from . import tables
from . import stats
from . import utils
from typing import Dict

# Spawn module-level logger
logger = logging.getLogger(__name__)

# Confidence Intervals
ONE_SIGMA_CI = scipy.special.erf(1/np.sqrt(2))
#    = 0.68268949213708585

# shorten type hint to save some space
Array = np.ndarray


def acf(z: Array, k: int) -> float:
    """Lag-k autocorrelation function.

    The autocorrelation function (ACF) is a fundamental way to describe a time
    series by multiplying it by a delayed version of itself, thereby showing
    the degree by which its value at one time is similar to its value at a
    certain later time.

    References:
         [RileyStable32]_ (5.5.3, pg.53-54)

    Args:
        z:  timeseries for which to calculate lag-k autocorrelation
        k:  lag interval

    Returns:
        timeseries autocorrelation factor at lag-k
    """

    # Mean value of the timeseries
    zbar = np.nanmean(z)
    z0 = z - zbar

    # Calculate autocorrelation factor
    r = np.nansum(z0[:-k]*z0[k:])/np.nansum(z0**2)

    return r


def acf_noise_id_core(z: Array, dmax: int, dmin: int = 0):
    """Core algorithm for the lag1 autocorrelation power law noise
    identification algorithm.

    References:
        [RileyStable32]_ (5.5.6 pg.56)
        http://www.stable32.com/Auto.pdf
        http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.503.9864&rep=rep1&type=pdf
        Power law noise identification using the lag 1 autocorrelation
        Riley,W.J. et al.
        18th European Frequency and Time Forum (EFTF 2004)
        https://ieeexplore.ieee.org/document/5075021

    Args:
        z:                  timeseries for which to calculate lag-1
                            autocorrelation. Input data should be at the
                            particular averaging time tau of interest.
        dmax:               maximum order of differencing
        dmin (optional):    minimum order of differencing


    Returns:
        an estimate p of the `alpha` of the dominant power law noise type.
    """

    d = 0
    done = False
    while not done:

        # Calculate lag-1 autocorrelation function
        r1 = acf(z=z, k=1)

        delta = r1/(1 + r1)

        if d >= dmin and (delta < 0.25 or d >= dmax):

            p = -round(2*delta) - 2*d
            done = True

        else:

            z = np.diff(z)
            d += 1

    return p


def acf_noise_id(data: Array, data_type: str, m: int, dev_type: str) -> int:
    """Effective method for identifying power law noise factor `alpha` at
    given averaging time.

    Noise identification based on Lag-1 autocorrelation function. Excellent
    discrimination for all common power law noises for both phase and
    frequency data, including difficult cases with mixed noises.

    References:
        [RileyStable32]_ (5.5.3-6 pg.53-7)

    Args:
        data:       array of input timeseries data. Before analysis, the data
                    should be preprocessed to remove outliers, discontinuities,
                    and  deterministic components
        data_type:  input data type. Either `phase` or `freq`.
        m:          averaging factor at which to estimate dominant noise type
        dev_type:   type of deviation used for analysis, e.g. `adev`.

    Returns:
        estimate of the `alpha` exponent, the dominant power law noise type.
    """

    # The input data should be for the particular averaging time, τ, of
    # interest, and it is therefore be necessary to decimate the phase
    # data or average the frequency data by the appropriate averaging factor
    # before applying the noise identification algorithm
    z = utils.decimate(data=data, m=m, data_type=data_type)

    # The dmax parameter should be set to 2 or 3 for an Allan or Hadamard (2
    # or 3-sample) variance analysis, respectively.
    dmax = tables.D_ORDER.get(dev_type, None)

    if dmax is None:
        raise KeyError(f"You provided an invalid {dev_type} "
                       f"dev_type for noise ID algorithm.")

    # Run lag1 autocorrelation noise id algorithm
    p = acf_noise_id_core(z=z, dmax=dmax)

    # The alpha result is equal to p+2 or p for phase or frequency data,
    # respectively, and may be rounded to an integer (although the fractional
    # part is useful for estimated mixed noises).
    alpha = p+2 if data_type == 'phase' else p

    return alpha


def noise_id(data: Array, data_type: str, m: int, tau: float,
             dev_type: str, n: int) -> int:
    """Noise identification pipeline, following what prescribed by Stable32.

    References:
        http://www.wriley.com/CI2.pdf
        [RileyStable32Manual]_ (Table pg. 97) (missing htotvar)
        [RileyStable32Manual]_ (Confidence Intervals pg. 89-93)
        [RileyStable32]_ (5.5.2-6 pg.53-7)
        Howe, Beard, Greenhall, Riley,
        A TOTAL ESTIMATOR OF THE HADAMARD FUNCTION USED FOR GPS OPERATIONS
        32nd PTTI, 2000
        https://apps.dtic.mil/dtic/tr/fulltext/u2/a484835.pdf

        # FIXME: find a reference for how to handle htotdev noiseid

    Args:
        data:       array of input timeseries data. Before analysis, the data
                    should be preprocessed to remove outliers, discontinuities,
                    and  deterministic components
        data_type:  input data type. Either `phase` or `freq`.
        m:          averaging factor at which to estimate dominant noise type
        tau:        corresponding averaging time
        dev_type:   type of deviation used for analysis, e.g. `adev`.
        n:          number of analysis points on which deviation was calculated

    Returns:
        estimate of the `alpha` exponent, the dominant power law noise type.
    """

    use_acf = ['adev', 'oadev', 'mdev', 'tdev', 'hdev', 'ohdev', 'totdev',
               'mtotdev', 'ttotdev', 'htotdev', 'theo1', 'theoh']
    use_b1 = ['adev', 'oadev', 'mdev', 'tdev', 'hdev', 'ohdev', 'totdev',
               'mtotdev', 'ttotdev', 'htotdev', 'theo1', 'theoh'] # hdev?
    use_rn = ['mdev', 'tdev', 'mtotdev', 'ttotdev', 'htotdev', 'theo1',
              'theoh']


    # Stable32 uses two methods for power law noise identification, based
    # respectively on the lag 1 autocorrelation and the B1 bias factor. The
    # former method is preferred, and is used when there are at least 30
    # analysis  data  points.

    # Size of dataset at given averaging factor:
    nn = utils.decimate(data=data, m=m, data_type=data_type).size

    if nn >= 30 and dev_type in use_acf:

        return acf_noise_id(data=data, data_type=data_type, m=m,
                            dev_type=dev_type)

    # Estimate alpha when there are less than 30 analysis datapoints (
    # acf_noise_1d throws this warning when it is the case)
    elif dev_type in use_b1:

        print(f"AF: {m} - Using B1 noise id")

        # B1 ratios expect phase data
        x = utils.input_to_phase(data=data, rate=m/tau, data_type=data_type)
        y = utils.phase2frequency(x=x, rate=m/tau)

        # compare b1 bias factor = standard variance / allan variance vs
        # expected value of this same ratio for pure noise types

        # Actual
        svar, _ = stats.calc_svar(x=x, m=m, tau=tau)
        avar, _ = stats.calc_avar(x=x, m=m, tau=tau)
        b1 = svar / avar
        #print(f"\tB1 ratio: {b1}")

        # B1 noise_id
        mu = b1_noise_id(measured=b1, N=n)  # this should be number of
        # frequency samples


        # If modified family of variances MVAR, TVAR or TOTMVAR
        # distinguish between WPM vs FPM by:
        # Supplement with R(n) ratio = mod allan / allan variance
        if mu == -2 and dev_type in use_rn:  # find if alpha = 1 or 2

            print("Using Rn ratio")

            # Actual
            mvar, _ = stats.calc_mvar(x=x, m=m, tau=tau)
            rn = mvar / avar

            # Rn noise_id
            alpha = rn_noise_id(measured=rn, m=m)
            return alpha

        # For the Hadamard variance, for which RRFM noise can apply (mu=3,
        # alpha=-4) the B1 ratio can be applied to frequency (rather than
        # phase) data, and adding 2 to the resulting mu
        elif m == 2:  # find if alpha = -3 or -4

            print("Using *B1 ratio")

            # *B1 ratio applies to frequency data
            y = data if data_type == 'freq' else utils.phase2frequency(x=data,
                                                                       rate=m/tau)
            svar, _ = stats.calc_svar_freq(y=y, m=m, tau=tau)
            avar, _ = stats.calc_avar_freq(y=x, m=m, tau=tau)
            b1star = svar / avar

            # B1 noise_id
            mu = b1_noise_id(measured=b1star, N=n)
            mu += 2

            assert mu <= 3, f"Invalid phase noise type mu: {mu}"


        # Get alpha value corresponding to identified mu

        alpha = [a for a, m in tables.ALPHA_TO_MU.items() if m == mu][0]

        #print(f"\tAssigned -> mu={mu} -> alpha={alpha}")
        return alpha

# -----

def get_error_bars(dev, m, tau, n, alpha=0, d=2, overlapping=True,
                   modified=False):
    """Calculate non-naive Allan deviation errors. Equivalent to Stable32.

    References:
        [RileyStable32]_ (5.3, pg.45-46)
        https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/20050061319.pdf

    Args:
        dev:                    raw deviation for which to compute error bars
        m:                      averaging factor at which deviation was
                                computed
        tau:                    averaging time  for which deviation was
                                computed
        n:                      number of samples on which deviation was
                                computed
        alpha (optional):       +2,...,-4 dominant noise type , either
                                estimated or known
        d (optional):           statistic code: 1 first-difference variance,
                                2 allan variance, 3 hadamard variance
        overlapping (optional): True if overlapping statistic used. False if
                                standard statistic used
        modified (optional):    True if modified statistic used. False if
                                standard statistic used.

    Returns:
        err_lo:                 non-naive lower 1-sigma error bar
        err_high:               non-naive higher 1-sigma error_bar
    """

    # Greenhalls EDF (Equivalent Degrees of Freedom)
    edf = edf_greenhall(alpha=alpha, d=d, m=m, N=n, overlapping=overlapping,
                        modified=modified)


    # with the known EDF we get CIs
    (lo, hi) = confidence_interval(dev=dev, edf=edf)

    err_lo = dev - lo
    err_hi = hi - dev

    return err_lo, err_hi


def confidence_interval(dev, edf, ci=ONE_SIGMA_CI):
    """ returns confidence interval (dev_min, dev_max)
        for a given deviation dev, equivalent degrees of freedom edf,
        and degree of confidence ci.

    Parameters
    ----------
    dev: float
        Mean value (e.g. adev) around which we produce the confidence interval
    edf: float
        Equivalent degrees of freedon
    ci: float, defaults to scipy.special.erf(1/math.sqrt(2)) for 1-sigma
    standard error set ci = scipy.special.erf(1/math.sqrt(2)) =
    0.68268949213708585

    Returns
    -------
    (dev_min, dev_max): (float, float)
        Confidence interval
    """
    ci_l = min(np.abs(ci), np.abs((ci-1))) / 2
    ci_h = 1 - ci_l

    # function from scipy, works OK, but scipy is large and slow to build
    chi2_l = scipy.stats.chi2.ppf(ci_l, edf)
    chi2_h = scipy.stats.chi2.ppf(ci_h, edf)

    variance = dev*dev
    var_l = float(edf) * variance / chi2_h  # NIST SP1065 eqn (45)
    var_h = float(edf) * variance / chi2_l

    return np.sqrt(var_l), np.sqrt(var_h)


def confidence_interval_noiseID(x, dev, af, dev_type="adev", data_type="phase", ci=ONE_SIGMA_CI):
    """ returns confidence interval (dev_min, dev_max)
        for a given deviation dev = Xdev( x, tau = af*(1/rate) )

        steps:
        1) identify noise type
        2) compute EDF
        3) compute confidence interval

    Parameters
    ----------
    x: numpy.array
        time-series
    dev: float
        Mean value (e.g. adev) around which we produce the confidence interval
    af: int
        averaging factor
    dev_type: string
        adev, oadev, mdev, tdev, hdev, ohdev
    data_type:
        "phase" or "freq"
    ci: float, defaults to scipy.special.erf(1/math.sqrt(2)) for 1-sigma
    standard error set ci = scipy.special.erf(1/math.sqrt(2)) =
    0.68268949213708585

    Returns
    -------
    (dev_min, dev_max): (float, float)
        Confidence interval
    """

    # 1) noise ID
    dmax = 2
    if (dev_type == "hdev") or (dev_type == "ohdev"):
        dmax = 3

    alpha_int = autocorr_noise_id(x, int(af), data_type=data_type, dmin=0,
                                  dmax=dmax)[0]

    # 2) EDF
    if dev_type == "adev":
        edf = edf_greenhall(alpha=alpha_int, d=2, m=af, N=len(x),
                            overlapping=False, modified=False)
    elif dev_type == "oadev":
        edf = edf_greenhall(alpha=alpha_int, d=2, m=af, N=len(x),
                            overlapping=True, modified=False)
    elif (dev_type == "mdev") or (dev_type == "tdev"):
        edf = edf_greenhall(alpha=alpha_int, d=2, m=af, N=len(x),
                            overlapping=True, modified=True)
    elif dev_type == "hdev":
        edf = edf_greenhall(alpha=alpha_int, d=3, m=af, N=len(x),
                            overlapping=False, modified=False)
    elif dev_type == "ohdev":
        edf = edf_greenhall(alpha=alpha_int, d=3, m=af, N=len(x),
                            overlapping=True, modified=False)
    else:
        raise NotImplementedError

    # 3) confidence interval
    (low, high) = confidence_interval(dev, edf, ci)

    return low, high


########################################################################
# Noise Identification using R(n)

# FIXME: get rid of this / update with new function contents
def rn(x, af, rate):
    """ R(n) ratio for noise identification

        ratio of MVAR to AVAR
    """

    (taus, devs, errs, ns) = allantools.adev(x, taus=[af*rate], data_type='phase', rate=rate)
    oadev_x = devs[0]
    (mtaus, mdevs, errs, ns) = allantools.mdev(x, taus=[af*rate], data_type='phase', rate=rate)
    mdev_x = mdevs[0]

    return pow(mdev_x/oadev_x, 2)


def rn_expected(m: int, alpha: int) -> float:
    """R(n) ratio expected from theory for given noise type and number of
    phase samples

    References:

        D. B. Sullivan, D. W. Allan, D. A. Howe, and F. L. Walls (editors)
        1990, “Characterization of Clocks and  Oscillators, ” National
        Institute of Standards and Technology  Technical Note 1337, Sec. A-6.
        Table 2.

    Args:
        m:      averaging factor
        alpha:  fractional frequency noise power-law exponent for which to
                calculate expected R(n) ratio

    Returns:
        R(n) ratio for given fractional frequency noise exponent
    """

    if alpha == 1:

        # assume measurement bandwidth `f_h` is 1/(2*tau_0)
        # then `w_h` == 2*pi*f_h = pi / tau_0
        # which means w_h*tau == w_h*m*tu_0 = pi * m

        return 3.37 / (1.04 + 3 * np.log(np.pi*m))

    elif alpha == 2:

        return 1. / m

    else:

        raise ValueError('Use B1 ratio instead for other noise types')



def rn_noise_id(measured: float, m: int) -> int:
    """ Identify fractional frequency noise power-law exponent type `alpha`
    from measured R(n) ratio.

    References:
        Howe, Beard, Greenhall, Riley,
        A TOTAL ESTIMATOR OF THE HADAMARD FUNCTION USED FOR GPS OPERATIONS
        32nd PTTI, 2000
        https://apps.dtic.mil/dtic/tr/fulltext/u2/a484835.pdf
        (Table 4, pg. 263)

    Args:
        measured:   measured R(n) ratio
        m:          averaging factor at which ratio was calculated

    Returns:
        identified `alpha` fractional frequency noise power-law exponent
    """

    alphas = [1, 2]

    # Expected R(n) ratio for each phase noise type
    d = {alpha: rn_expected(m=m, alpha=alpha) for alpha in alphas}

    # Boundary between FLPM (alpha=1) and WHPM (alpha=2)
    bndry = np.sqrt(d[1] * d[2])

    # Compare measured result against boundary
    alpha = 1 if measured > bndry else 2

    return alpha


# TODO: check if calculation is equivalent / better and then get rid
def rn_theory(af, b):
    """R(n) ratio expected from theory for given noise type and number of
    phase samples

        alpha = beta + 2

    References:

        D. B. Sullivan, D. W. Allan, D. A. Howe, and F. L. Walls (editors)
        1990, “Characterization of Clocks and  Oscillators, ” National
        Institute of Standards and Technology  Technical Note 1337, Sec. A-6.
        Table 2.
    """

    # From IEEE1139-2008
    #   alpha   beta    ADEV_mu MDEV_mu Rn_mu
    #   -2      -4       1       1       0      Random Walk FM
    #   -1      -3       0       0       0      Flicker FM
    #    0      -2      -1      -1       0      White FM
    #    1      -1      -2      -2       0      Flicker PM
    #    2      0       -2      -3      -1      White PM

    # (a=-3 flicker walk FM)
    # (a=-4 random run FM)
    if b == 0:
        return pow(af, -1)
    elif b == -1:
        # f_h = 0.5/tau0  (assumed!)
        # af = tau/tau0
        # so f_h*tau = 0.5/tau0 * af*tau0 = 0.5*af
        avar = (1.038+3*np.log(2*np.pi*0.5*af)) / (4.0*pow(np.pi, 2))
        mvar = 3*np.log(256.0/27.0)/(8.0*pow(np.pi, 2))
        return mvar/avar
    else:
        return pow(af, 0)


# TODO: get rid
def rn_boundary(af, b_hi):
    """
    R(n) ratio boundary for selecting between [b_hi-1, b_hi]
    alpha = b + 2
    """
    return np.sqrt(rn_theory(af, b_hi)*rn_theory(af, b_hi-1))  # geometric mean

########################################################################
# Noise Identification using B1

# FIXME: get rid of this / update with new function contents
def b1_old(x, af, rate):
    """ B1 ratio for noise identification
        (and bias correction?)

        ratio of Standard Variace to AVAR

        Howe, Beard, Greenhall, Riley,
        A TOTAL ESTIMATOR OF THE HADAMARD FUNCTION USED FOR GPS OPERATIONS
        32nd PTTI, 2000
        https://apps.dtic.mil/dtic/tr/fulltext/u2/a484835.pdf

        [Barnes1974]_
        https://tf.nist.gov/general/pdf/11.pdf
    """
    #(taus, devs, errs, ns) = allantools.adev(x, taus=[af*rate],
    # data_type="phase", rate=rate)
    #oadev_x = devs[0]
    #avar = pow(oadev_x, 2.0)
    avar  = 1

    # variance of y, at given af
    y = np.diff(x)
    y_cut = np.array(y[:len(y)-(len(y)%af)]) # cut to length
    assert len(y_cut)%af == 0
    y_shaped = y_cut.reshape((int(len(y_cut)/af), af))
    y_averaged = np.average(y_shaped, axis=1) # average
    var = np.var(y_averaged, ddof=1)

    return var/avar


def b1_expected(N: int, mu: int, r: float = 1) -> float:
    """ Expected B1 ratio for a timeseries of N fractional frequency
    datapoints and phase noise exponent mu

    References:
        Howe, Beard, Greenhall, Riley,
        A TOTAL ESTIMATOR OF THE HADAMARD FUNCTION USED FOR GPS OPERATIONS
        32nd PTTI, 2000
        https://apps.dtic.mil/dtic/tr/fulltext/u2/a484835.pdf
        (Table 3, pg. 261-2)

    Args:
        N:  number of fractional frequency datapoints. This is M-1,
            for M phase data points.
        mu: phase noise power-law exponent for which to calculate expected
            B1 ratio
        r:  dead time ratio. Set to 1.

    Returns:
        B1 ratio for given phase noise exponent.
    """

    if mu == 2:  # FWFM
        return N * (N + 1) / 6.0

    elif mu == 1:  # RWFM
        return N / 2.

    elif mu == 0:  # FLFM
        return N*np.log(N) / (2*(N-1)*np.log(2))

    elif mu == -1:  # WHFM
        return 1

    elif mu == -2:  # WHPM or FLPM
        return (N**2 - 1) / (1.5*N*(N-1.))

    else:
        logger.warning("Calculating B1 ratio for phase noise exponent outside "
                       "bounds")
        return N * (1 - N**mu) / (2 * (N-1) * (1 - 2**mu))


def b1_noise_id(measured: float, N: int, r: float = 1) -> int:
    """ Identify phase noise power-law exponent type `mu` from measured B1
    ratio.

    References:
        Howe, Beard, Greenhall, Riley,
        A TOTAL ESTIMATOR OF THE HADAMARD FUNCTION USED FOR GPS OPERATIONS
        32nd PTTI, 2000
        https://apps.dtic.mil/dtic/tr/fulltext/u2/a484835.pdf
        (Table 4, pg. 263)

    Args:
        measured:   measured B1 ratio
        N:          number of fractional frequency datapoints on which
                    measured B1 ratio was calculated. This is M-1,
                    for M phase data points.
        r:          dead time ratio. Set to 1.

    Returns:
        identified `mu` phase noise power-law exponent
    """

    mus = [2, 1, 0, -1, -2]

    # Expected B1 ratio for each phase noise type
    d = {mu: b1_expected(N=N, mu=mu, r=r) for mu in mus}

    # Boundaries between noise types
    b = {}
    for mu, b1 in d.items():

        if mu > -2:

            bndry = 0.5*(b1 + d[mu-1]) if mu == 2 else np.sqrt(b1 * d[mu-1])
            b[mu] = bndry


    #print(f"\tBoundaries {b}")

    # Assign measured b1 to most plausible noise type:
    # the actual measured ratio is tested against mu values downwards from
    # the largest applicable mu
    for mu, bndry in b.items():

        if measured > bndry:

            return mu

    # If the above didn't return anything, give lowest possible noise type
    return -2

# TODO: get rid
def b1_boundary(b_hi, N):
    """
    B1 ratio boundary for selecting between [b_hi-1, b_hi]
    alpha = b + 2
    """
    b_lo = b_hi-1
    b1_lo = b1_expected(N, b_to_mu(b_lo))
    b1_hi = b1_expected(N, b_to_mu(b_hi))
    if b1_lo >= -4:
        return np.sqrt(b1_lo*b1_hi)  # geometric mean
    else:
        return 0.5*(b1_lo+b1_hi)  # arithemtic mean


def b_to_mu(b):
    """
    return mu, parameter needed for B1 ratio function b1()
    alpha = b + 2
    """
    a = b + 2
    if a == +2:
        return -2
    elif a == +1:
        return -2
    elif a == 0:
        return -1
    elif a == -1:
        return 0
    elif a == -2:
        return 1
    elif a == -3:
        return 2
    elif a == -4:
        return 3
    assert False

########################################################################
# Noise Identification using ACF


def lag1_acf(x, detrend_deg=1):
    """ Lag-1 autocorrelation function
        as defined in Riley 2004, Eqn (2)
        used by autocorr_noise_id()

        Parameters
        ----------
        x: numpy.array
            time-series
        Returns
        -------
        ACF: float
            Lag-1 autocorrelation for input time-series x

        Notes
        -----
        * a faster algorithm based on FFT might be better!?
        * numpy.corrcoeff() gives similar but not identical results.
            #c = np.corrcoef( np.array(x[:-lag]), np.array(x[lag:]) )
            #r1 = c[0,1] # lag-1 autocorrelation of x
    """
    mu = np.mean(x)
    a = 0
    b = 0
    for n in range(len(x)-1):
        a = a + (x[n]-mu)*(x[n+1]-mu)
    # for n in range(len(x)):
    for xn in x:
        b = b+pow(xn-mu, 2)
    return a/b


def autocorr_noise_id(x, af, data_type="phase", dmin=0, dmax=2):
    """ Lag-1 autocorrelation based noise identification

    Parameters
    ----------
    x: numpy.array
        phase or fractional frequency time-series data
        minimum recommended length is len(x)>30 roughly.
    af: int
        averaging factor
    data_type: string {'phase', 'freq'}
        "phase" for phase data in seconds
        "freq" for fractional frequency data
    dmin: int
        minimum required number of differentiations in the algorithm
    dmax: int
        maximum number of differentiations
        defaults to 2 for ADEV
        set to 3 for HDEV

    Returns
    -------
    alpha_int: int
        noise-slope as integer
    alpha: float
        noise-slope as float
    d: int
        number of differentiations of the time-series performed

    Notes
    -----
        http://www.stable32.com/Auto.pdf
        http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.503.9864&rep=rep1&type=pdf

    Power law noise identification using the lag 1 autocorrelation
    Riley,W.J. et al.
    18th European Frequency and Time Forum (EFTF 2004)
    https://ieeexplore.ieee.org/document/5075021

    """
    d = 0 # number of differentiations
    lag = 1
    if data_type == "phase":
        if af > 1:
            # x = scipy.signal.decimate(x, af, n=1, ftype='fir')
            x = x[0:len(x):af] # decimate by averaging factor
        x = detrend(x, deg=2) # remove quadratic trend (frequency offset and drift)
    elif data_type == "freq":
        # average by averaging factor
        y_cut = np.array(x[:len(x)-(len(x)%af)]) # cut to length
        assert len(y_cut) % af == 0
        y_shaped = y_cut.reshape((int(len(y_cut)/af), af))
        x = np.average(y_shaped, axis=1) # average
        x = detrend(x, deg=1)  # remove frequency drift

    # require minimum length for time-series
    if len(x) < 30:
        print("autocorr_noise_id() Don't know how to do noise-ID for time-series length= %d"%len(x))
        raise NotImplementedError

    while True:
        r1 = lag1_acf(x)
        rho = r1/(1.0+r1)
        if d >= dmin and (rho < 0.25 or d >= dmax):
            p = -2*(rho+d)
            phase_add2 = 0
            if data_type == "phase":
                phase_add2 = 2
            alpha = p+phase_add2
            alpha_int = int(-1.0*np.round(2*rho) - 2.0*d) + phase_add2
            #print "d=",d,"alpha=",p+2
            return alpha_int, alpha, d, rho
        else:
            x = np.diff(x)
            d = d + 1

########################################################################
# Equivalent Degrees of Freedom


def edf_greenhall_simple(alpha, d, m, S, F, N):
    """ Eqn (13) from [Greenhall2004]_ """
    L = m/F+m*d # length of filter applied to phase samples
    M = 1 + np.floor(S*(N-L) / m)
    J = min(M, (d+1)*S)
    inv_edf = (1.0/(pow(greenhall_sz(0, F, alpha, d), 2)*M)) * greenhall_BasicSum(J, M, S, F, alpha, d)
    return 1.0/inv_edf


def edf_greenhall(alpha, d, m, N, overlapping=False, modified=False, verbose=False):
    """ returns Equivalent degrees of freedom

        Parameters
        ----------
        alpha: int
            noise type, +2...-4
        d: int
            1 first-difference variance
            2 Allan variance
            3 Hadamard variance
            require alpha+2*d>1
        m: int
            averaging factor
            tau = m*tau0 = m*(1/rate)
        N: int
            number of phase observations (length of time-series)
        overlapping: bool
            True for oadev, ohdev
        modified: bool
            True for mdev, tdev

        Returns
        -------
        edf: float
            Equivalent degrees of freedom

        Greenhall, Riley, 2004
        https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/20050061319.pdf
        UNCERTAINTY OF STABILITY VARIANCES BASED ON FINITE DIFFERENCES

        Notes
        -----
        Used for the following deviations (see http://www.wriley.com/CI2.pdf page 8)
        adev()
        oadev()
        mdev()
        tdev()
        hdev()
        ohdev()
    """

    if modified:
        F = 1 # F filter factor, 1 modified variance, m unmodified variance
    else:
        F = int(m)
    if overlapping: # S stride factor, 1 nonoverlapped estimator,
        S = int(m)  # m overlapped estimator (estimator stride = tau/S )
    else:
        S = 1
    assert(alpha+2*d > 1.0)
    L = m/F+m*d # length of filter applied to phase samples
    M = 1 + np.floor(S*(N-L) / m)
    J = min(M, (d+1)*S)
    J_max = 100
    r = M/S
    if int(F) == 1 and modified: # case 1, modified variances, all alpha
        if J <= J_max:
            inv_edf = (1.0/(pow(greenhall_sz(0, 1, alpha, d), 2)*M))* \
                       greenhall_BasicSum(J, M, S, 1, alpha, d)
            if verbose:
                print("case 1.1 edf= %3f" % float(1.0/inv_edf))
            return 1.0/inv_edf
        elif r > d+1:
            (a0, a1) = greenhall_table1(alpha, d)
            inv_edf = (1.0/r)*(a0-a1/r)
            if verbose:
                print("case 1.2 edf= %3f" % float(1.0/inv_edf))
            return 1.0/inv_edf
        else:
            m_prime = J_max/r
            inv_edf = (1.0/(pow(greenhall_sz(0, F, alpha, d), 2)*J_max))* \
                       greenhall_BasicSum(J_max, J_max, m_prime, 1, alpha, d)
            if verbose:
                print("case 1.3 edf= %3f" % float(1.0/inv_edf))
            return 1.0/inv_edf
    elif int(F) == int(m) and int(alpha) <= 0 and not modified: 
        # case 2, unmodified variances, alpha <= 0
        if J <= J_max:
            if m*(d+1) <= J_max:
                m_prime = m
                variant = "a"
            else:
                m_prime = float('inf')
                variant = "b"

            inv_edf = (1.0/(pow(greenhall_sz(0, m_prime, alpha, d), 2)*M))* \
                       greenhall_BasicSum(J, M, S, m_prime, alpha, d)
            if verbose:
                print("case 2.1%s edf= %3f" % (variant, float(1.0/inv_edf)))
            return 1.0/inv_edf
        elif r > d+1:
            (a0, a1) = greenhall_table2(alpha, d)
            inv_edf = (1.0/r)*(a0-a1/r)
            if verbose:
                print("case 2.2 edf= %3f" % float(1.0/inv_edf))
            return 1.0/inv_edf
        else:
            m_prime = J_max/r
            inv_edf = (1.0/(pow(greenhall_sz(0, float('inf'), alpha, d), 2)*J_max))* \
                       greenhall_BasicSum(J_max, J_max, m_prime, float('inf'), alpha, d)
            if verbose:
                print("case 2.3 edf= %3f" % float(1.0/inv_edf))
            return 1.0/inv_edf
    elif int(F) == int(m) and int(alpha) == 1 and not modified:
        # case 3, unmodified variances, alpha=1
        if J <= J_max:
            inv_edf = (1.0/(pow(greenhall_sz(0, m, 1, d), 2)*M))* \
                       greenhall_BasicSum(J, M, S, m, 1, d) # note: m<1e6 to avoid roundoff
            if verbose:
                print("case 3.1 edf= %3f" % float(1.0/inv_edf))
            return 1.0/inv_edf
        elif r > d+1:
            (a0, a1) = greenhall_table2(alpha, d)
            (b0, b1) = greenhall_table3(alpha, d)
            inv_edf = (1.0/(pow(b0+b1*np.log(m), 2)*r))*(a0-a1/r)
            if verbose:
                print("case 3.2 edf= %3f" % float(1.0/inv_edf))
            return 1.0/inv_edf
        else:
            m_prime = J_max/r
            (b0, b1) = greenhall_table3(alpha, d)
            inv_edf = (1.0/(pow(b0+b1*np.log(m), 2)*J_max))* \
                       greenhall_BasicSum(J_max, J_max, m_prime, m_prime, 1, d)
            if verbose:
                print("case 3.3 edf= %3f" % float(1.0/inv_edf))
            return 1.0/inv_edf
    elif int(F) == int(m) and int(alpha) == 2 and not modified:
        # case 4, unmodified variances, alpha=2
        K = np.ceil(r)
        if K <= d:
            raise NotImplementedError  # FIXME: add formula from the paper here!
        else:
            a0 = scipy.special.binom(4*d, 2*d) / pow(scipy.special.binom(2*d, d), 2)
            a1 = d/2.0
            inv_edf = (1.0/M)*(a0-a1/r)
            if verbose:
                print("case 4.2 edf= %3f" % float(1.0/inv_edf))
            return 1.0/inv_edf

    print("greenhall_edf() no matching case!")

    raise NotImplementedError


def greenhall_BasicSum(J, M, S, F, alpha, d):
    """ Eqn (10) from [Greenhall2004]_ """
    first = pow(greenhall_sz(0, F, alpha, d), 2)
    second = (1-float(J)/float(M))*pow(greenhall_sz(float(J)/float(S), F, alpha, d), 2)
    third = 0
    for j in range(1, int(J)):
        third += 2*(1.0-float(j)/float(M))*pow(greenhall_sz(float(j)/float(S), F, alpha, d), 2)
    return first+second+third


def greenhall_sz(t, F, alpha, d):
    """ Eqn (9) from [Greenhall2004]_ """
    if d == 1:
        a = 2*greenhall_sx(t, F, alpha)
        b = greenhall_sx(t-1.0, F, alpha)
        c = greenhall_sx(t+1.0, F, alpha)
        return a-b-c
    elif d == 2:
        a = 6*greenhall_sx(t, F, alpha)
        b = 4*greenhall_sx(t-1.0, F, alpha)
        c = 4*greenhall_sx(t+1.0, F, alpha)
        dd = greenhall_sx(t-2.0, F, alpha)
        e = greenhall_sx(t+2.0, F, alpha)
        return a-b-c+dd+e
    elif d == 3:
        a = 20.0*greenhall_sx(t, F, alpha)
        b = 15.0*greenhall_sx(t-1.0, F, alpha)
        c = 15.0*greenhall_sx(t+1.0, F, alpha)
        dd = 6.0*greenhall_sx(t-2.0, F, alpha)
        e = 6.0*greenhall_sx(t+2.0, F, alpha)
        f = greenhall_sx(t-3.0, F, alpha)
        g = greenhall_sx(t+3.0, F, alpha)
        return a-b-c+dd+e-f-g

    assert 0  # ERROR


def greenhall_sx(t, F, alpha):
    """ Eqn (8) from [Greenhall2004]_
    """
    if F == float('inf'):
        return greenhall_sw(t, alpha+2)
    a = 2*greenhall_sw(t, alpha)
    b = greenhall_sw(t-1.0/float(F), alpha)
    c = greenhall_sw(t+1.0/float(F), alpha)

    return pow(F, 2)*(a-b-c)


def greenhall_sw(t, alpha):
    """ Eqn (7) from [Greenhall2004]_
    """
    alpha = int(alpha)
    if alpha == 2:
        return -np.abs(t)
    elif alpha == 1:
        if t == 0:
            return 0
        else:
            return pow(t, 2)*np.log(np.abs(t))
    elif alpha == 0:
        return np.abs(pow(t, 3))
    elif alpha == -1:
        if t == 0:
            return 0
        else:
            return pow(t, 4)*np.log(np.abs(t))
    elif alpha == -2:
        return np.abs(pow(t, 5))
    elif alpha == -3:
        if t == 0:
            return 0
        else:
            return pow(t, 6)*np.log(np.abs(t))
    elif alpha == -4:
        return np.abs(pow(t, 7))

    assert 0  # ERROR


def greenhall_table3(alpha, d):
    """ Table 3 from Greenhall 2004 """
    assert(alpha == 1)
    idx = d-1
    table3 = [(6.0, 4.0), (15.23, 12.0), (47.8, 40.0)]
    return table3[idx]


def greenhall_table2(alpha, d):
    """ Table 2 from Greenhall 2004 """
    row_idx = int(-alpha+2) # map 2-> row0 and -4-> row6
    assert(row_idx in [0, 1, 2, 3, 4, 5])
    col_idx = int(d-1)
    table2 = [[(3.0/2.0, 1.0/2.0), (35.0/18.0, 1.0), (231.0/100.0, 3.0/2.0)],  # alpha=+2
              [(78.6, 25.2), (790.0, 410.0), (9950.0, 6520.0)],
              [(2.0/3.0, 1.0/6.0), (2.0/3.0, 1.0/3.0), (7.0/9.0, 1.0/2.0)],  # alpha=0
              [(-1, -1), (0.852, 0.375), (0.997, 0.617)],  # -1
              [(-1, -1), (1.079, 0.368), (1.033, 0.607)],  # -2
              [(-1, -1), (-1, -1), (1.053, 0.553)],  # -3
              [(-1, -1), (-1, -1), (1.302, 0.535)],  # alpha=-4
              ]

    # print("table2 = ", table2[row_idx][col_idx])
    return table2[row_idx][col_idx]


def greenhall_table1(alpha, d):
    """ Table 1 from Greenhall 2004 """
    row_idx = int(-alpha+2) # map 2-> row0 and -4-> row6
    col_idx = int(d-1)
    table1 = [[(2.0/3.0, 1.0/3.0), (7.0/9.0, 1.0/2.0), (22.0/25.0, 2.0/3.0)],  # alpha=+2
              [(0.840, 0.345), (0.997, 0.616), (1.141, 0.843)],
              [(1.079, 0.368), (1.033, 0.607), (1.184, 0.848)],
              [(-1, -1), (1.048, 0.534), (1.180, 0.816)],  # -1
              [(-1, -1), (1.302, 0.535), (1.175, 0.777)],  #-2
              [(-1, -1), (-1, -1), (1.194, 0.703)],  #-3
              [(-1, -1), (-1, -1), (1.489, 0.702)],  # alpha=-4
              ]

    # print("table1 = ", table1[row_idx][col_idx])
    return table1[row_idx][col_idx]


def edf_totdev(N, m, alpha):
    """ Equivalent degrees of freedom for Total Deviation
        FIXME: what is the right behavior for alpha outside 0,-1,-2?

        NIST SP1065 page 41, Table 7
    """
    alpha = int(alpha)
    if alpha in [0, -1, -2]:
        # alpha  0 WFM
        # alpha -1 FFM
        # alpha -2 RWFM
        NIST_SP1065_table7 = [(1.50, 0.0), (1.17, 0.22), (0.93, 0.36)]
        (b, c) = NIST_SP1065_table7[int(abs(alpha))]
        return b*(float(N)/float(m))-c
    # alpha outside 0, -1, -2:
    return edf_simple(N, m, alpha)


def edf_mtotdev(N, m, alpha):
    """ Equivalent degrees of freedom for Modified Total Deviation

        NIST SP1065 page 41, Table 8
    """
    assert(alpha in [2, 1, 0, -1, -2])
    NIST_SP1065_table8 = [(1.90, 2.1), (1.20, 1.40), (1.10, 1.2), (0.85, 0.50), (0.75, 0.31)]
    # (b, c) = NIST_SP1065_table8[ abs(alpha-2) ]
    (b, c) = NIST_SP1065_table8[abs(alpha-2)]
    edf = b*(float(N)/float(m))-c
    print("mtotdev b,c= ", (b, c), " edf=", edf)
    return edf


def edf_simple(N, m, alpha):
    """Equivalent degrees of freedom.
    Simple approximate formulae.

    Parameters
    ----------
    N : int
        the number of phase samples
    m : int
        averaging factor, tau = m * tau0
    alpha: int
        exponent of f for the frequency PSD:
        'wp' returns white phase noise.             alpha=+2
        'wf' returns white frequency noise.         alpha= 0
        'fp' returns flicker phase noise.           alpha=+1
        'ff' returns flicker frequency noise.       alpha=-1
        'rf' returns random walk frequency noise.   alpha=-2
        If the input is not recognized, it defaults to idealized, uncorrelated
        noise with (N-1) degrees of freedom.

    Notes
    -----
       S. Stein, Frequency and Time - Their Measurement and
       Characterization. Precision Frequency Control Vol 2, 1985, pp 191-416.
       http://tf.boulder.nist.gov/general/pdf/666.pdf

    Returns
    -------
    edf : float
        Equivalent degrees of freedom

    """

    edf = (N - 1)  # default value

    N = float(N)
    m = float(m)
    if alpha in [2, 1, 0, -1, -2]:
        # NIST SP 1065, Table 5
        if alpha == +2:
            edf = (N + 1) * (N - 2*m) / (2 * (N - m))

        if alpha == 0:
            edf = (((3 * (N - 1) / (2 * m)) - (2 * (N - 2) / N)) *
                   ((4*pow(m, 2)) / ((4*pow(m, 2)) + 5)))

        if alpha == 1:
            a = (N - 1)/(2 * m)
            b = (2 * m + 1) * (N - 1) / 4
            edf = np.exp(np.sqrt(np.log(a) * np.log(b)))

        if alpha == -1:
            if m == 1:
                edf = 2 * (N - 2) /(2.3 * N - 4.9)
            if m >= 2:
                edf = 5 * N**2 / (4 * m * (N + (3 * m)))

        if alpha == -2:
            a = (N - 2) / (m * (N - 3)**2)
            b = (N - 1)**2
            c = 3 * m * (N - 1)
            d = 4 * m **2
            edf = a * (b - c + d)

    else:
        print("Noise type not recognized. Defaulting to N - 1 degrees of freedom.")

    return edf

########################################################################
# end of ci.py
