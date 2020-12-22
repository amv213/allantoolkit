"""
this file is part of allantoolkit, https://github.com/aewallin/allantools

- functions for confidence intervals
- functions for computing equivalent degrees of freedom
"""

import logging
import numpy as np
import scipy.special
import scipy.stats  # used in confidence_intervals()
import scipy.signal  # decimation in lag-1 acf
from . import tables
from . import utils


# Spawn module-level logger
logger = logging.getLogger(__name__)

# Confidence Intervals
ONE_SIGMA_CI = scipy.special.erf(1/np.sqrt(2))
#    = 0.68268949213708585

# shorten type hint to save some space
Array = np.ndarray


def get_error_bars(x: Array, m: int, dev: float, n: int, alpha: int,
                   dev_type: str):
    """Calculate non-naive Allan deviation errors. Equivalent to Stable32.

    References:
        [RileyStable32Manual]_ (Confidence Intervals, pg.89)
        http://www.wriley.com/CI2.pdf
        https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/20050061319.pdf

    Args:
        x:          phase data from which deviation was computed, in units of
                    seconds.
        m:          averaging factor at which deviation was computed
        dev:        deviation value for which to compute error bars
        n:          number of analysis samples used to compute deviation
        alpha:      dominant power law frequency noise type
        dev_type:   type of deviation for which error bars are being
                    calculated, e.g. `adev`.

    Returns:
        err_lo:                 non-naive lower 1-sigma error bar
        err_high:               non-naive higher 1-sigma error_bar
    """

    # Chi-squared statistics can be applied to calculate single and
    # double-sided confidence intervals at any desired confidence factor.
    # Those calculations, based on a determination of the number of degrees of
    # freedom for the estimated noise type, are the confidence intervals used
    # by the Stable32 variance functions.

    # Dispatch to appropriate edf calculator for given dev_type
    # should have function signature func(x, m, alpha) -> edf
    try:
        edf_func = globals()['calc_edf_' + dev_type]
    except KeyError:
        raise ValueError(f"Error bar calculation for {dev_type} has not "
                         f"been implemented.")

    edf = edf_func(x=x, m=m, alpha=alpha)
    edf = round(edf, 3)

    print(f"{dev_type.upper()} | AF: {m} | EDF {edf}")

    # with the known EDF we get CIs
    (lo, hi) = confidence_interval(dev=dev, edf=edf)

    print(f"\tMIN ADEV {lo} | DEV {dev} | MAX ADEV {hi}")

    # From CIs we get error bars
    err_lo = dev - lo
    err_hi = hi - dev

    return err_lo, err_hi


# Dispatchers

def calc_edf_adev(x: Array, m: int, alpha: int) -> float:
    """Calculate equivalent number of Chi-squared degrees of freedom (edf)
    for ADEV

    References:
        C. Greenhall and W. Riley, "Uncertainty of Stability Variances Based
        on Finite Differences", Proc. 2003 PTTI Meeting , December 2003

        http://www.wriley.com/CI2.pdf

    Args:
        x:          phase data from which deviation was computed, in units of
                    seconds.
        m:          averaging factor at which deviation was computed
        alpha:      dominant power law frequency noise type

    Returns:
        equivalent number of Chi-squared degrees of freedom (edf)
    """

    # TODO: get d, overlapping, and modified from a table

    return cedf(alpha=alpha, d=2, m=m, N=int(x.size), overlapping=False,
                modified=False)


def calc_edf_oadev(x: Array, m: int, alpha: int) -> float:
    """Calculate equivalent number of Chi-squared degrees of freedom (edf)
    for OADEV

    References:
        C. Greenhall and W. Riley, "Uncertainty of Stability Variances Based
        on Finite Differences", Proc. 2003 PTTI Meeting , December 2003

        http://www.wriley.com/CI2.pdf

    Args:
        x:          phase data from which deviation was computed, in units of
                    seconds.
        m:          averaging factor at which deviation was computed
        alpha:      dominant power law frequency noise type

    Returns:
        equivalent number of Chi-squared degrees of freedom (edf)
    """

    # TODO: get d, overlapping, and modified from a table

    return cedf(alpha=alpha, d=2, m=m, N=int(x.size), overlapping=True,
                modified=False)


def calc_edf_mdev(x: Array, m: int, alpha: int) -> float:
    """Calculate equivalent number of Chi-squared degrees of freedom (edf)
    for MDEV.

    References:
        C. Greenhall and W. Riley, "Uncertainty of Stability Variances Based
        on Finite Differences", Proc. 2003 PTTI Meeting , December 2003

        http://www.wriley.com/CI2.pdf

    Args:
        x:          phase data from which deviation was computed, in units of
                    seconds.
        m:          averaging factor at which deviation was computed
        alpha:      dominant power law frequency noise type

    Returns:
        equivalent number of Chi-squared degrees of freedom (edf)
    """

    # TODO: get d, overlapping, and modified from a table

    return cedf(alpha=alpha, d=2, m=m, N=int(x.size), overlapping=True,
                modified=True)


def calc_edf_tdev(x: Array, m: int, alpha: int) -> float:
    """Calculate equivalent number of Chi-squared degrees of freedom (edf)
    for TDEV.

    References:
        C. Greenhall and W. Riley, "Uncertainty of Stability Variances Based
        on Finite Differences", Proc. 2003 PTTI Meeting , December 2003

        http://www.wriley.com/CI2.pdf

    Args:
        x:          phase data from which deviation was computed, in units of
                    seconds.
        m:          averaging factor at which deviation was computed
        alpha:      dominant power law frequency noise type

    Returns:
        equivalent number of Chi-squared degrees of freedom (edf)
    """

    # TODO: get d, overlapping, and modified from a table

    return cedf(alpha=alpha, d=2, m=m, N=int(x.size), overlapping=True,
                modified=True)


def calc_edf_hdev(x: Array, m: int, alpha: int) -> float:
    """Calculate equivalent number of Chi-squared degrees of freedom (edf)
    for HDEV.

    References:
        C. Greenhall and W. Riley, "Uncertainty of Stability Variances Based
        on Finite Differences", Proc. 2003 PTTI Meeting , December 2003

        http://www.wriley.com/CI2.pdf

    Args:
        x:          phase data from which deviation was computed, in units of
                    seconds.
        m:          averaging factor at which deviation was computed
        alpha:      dominant power law frequency noise type

    Returns:
        equivalent number of Chi-squared degrees of freedom (edf)
    """

    # TODO: get d, overlapping, and modified from a table

    return cedf(alpha=alpha, d=3, m=m, N=int(x.size), overlapping=False,
                modified=False)


def calc_edf_ohdev(x: Array, m: int, alpha: int) -> float:
    """Calculate equivalent number of Chi-squared degrees of freedom (edf)
    for OHDEV.

    References:
        C. Greenhall and W. Riley, "Uncertainty of Stability Variances Based
        on Finite Differences", Proc. 2003 PTTI Meeting , December 2003

        http://www.wriley.com/CI2.pdf

    Args:
        x:          phase data from which deviation was computed, in units of
                    seconds.
        m:          averaging factor at which deviation was computed
        alpha:      dominant power law frequency noise type

    Returns:
        equivalent number of Chi-squared degrees of freedom (edf)
    """

    # TODO: get d, overlapping, and modified from a table

    return cedf(alpha=alpha, d=3, m=m, N=int(x.size), overlapping=True,
                modified=False)

# Combined Greenhall EDF algorithm

def cedf(alpha: int, d: int, m: int, N: int, overlapping: bool = False,
         modified: bool = False, version: str = 'full') -> float:
    """ Returns combined edf value, as per Greenhall & Riley PTTI '03.

    This method is based on modeling the phase as the first difference of a
    continuous-time pure power-law process.

    References:

        C. Greenhall and W. Riley, "Uncertainty of Stability Variances Based
        on Finite Differences", Proc. 2003 PTTI Meeting , December 2003

    Args:
        alpha:          frequency power law noise exponent.
        d:              order of phase difference of the variance for which
                        to calculate edf.
        m:              averaging factor at which to calculate edf.
        modified:       `True` if calculating edf for a `modified` variance.
                        `False` otherwise.
        overlapping:    `True` if calculating edf for an `overlapping`
                        variance. `False` otherwise.
        N:              number of phase data points, with sample period tau_0.
        version:        `simple` to only calculate one step of the algorithm.
                        Defaults to `full`.

    Returns:
        equivalent degrees of freedom of the variance estimator at given
        averaging factor.

    Notes:
        Used by Stable32 for AVAR, MVAR, TVAR, HVAR, and their overlapping
        versions. Output edf result matches Stable32 `Sigma` tool (need to
        tick the ACF noise id box)

    """

    # Consistency check on input params
    if alpha + 2*d < 1:
        raise ValueError(f"A d = {d} deviation cannot discriminate noise types"
                         f" alpha = {alpha}. Update input parameters.")

    # Set filter factor
    F = 1 if modified else m

    # Set stride factor (estimator stride = tau / S)
    S = m if overlapping else 1

    # Set an integer constant
    J_max = 100  # suggested value

    # 3.4. INITIAL STEPS

    # 1. compute M, the number of summands in the estimator

    L = m//F + m*d  # length of filter applied to the phase samples

    if N >= L:

        M = 1 + int(np.floor(S*(N-L)/m))

    else:
        raise ValueError(f"Not enough data to calculate edf. Should have N "
                         f">= {L}, but only have {N} samples.")

    # 2. Let
    J = int(min(M, (d+1)*S))

    # 3.5 Main procedure, simplified version
    if version == 'simple':

        inv_edf = cedf_basic_sum(J=J, M=M, S=S, F=F, alpha=alpha, d=d) / \
                      (M * cedf_sz(t=0, F=F, alpha=alpha, d=d)**2)

        return 1 / inv_edf

    # 3.6 Main Procedure, full version

    r = M//S

    # There are four cases. The calculations use coefficients from three
    # numerical tables:

    #  3.6.1 Case 1. Modified variances: F = 1, all alpha.
    #  This case also applies to unmodified variances when F = m = 1

    if F == 1:

        if J <= J_max:

            inv_edf = cedf_basic_sum(J=J, M=M, S=S, F=1, alpha=alpha, d=d) / \
                      (M * cedf_sz(t=0, F=1, alpha=alpha, d=d)**2)

        elif J > J_max and r >= d+1:

            a0, a1 = tables.GREENHALL_TABLE1[alpha][d]

            inv_edf = (a0 - a1/r) / r

        else:

            m_prime = J_max / r  # not necessarily an integer

            inv_edf = cedf_basic_sum(
                J=J_max, M=J_max, S=m_prime, F=1, alpha=alpha, d=d
            ) / (J_max * cedf_sz(t=0, F=1, alpha=alpha, d=d)**2)

    #  3.6.2 Case 2. Unmodified variances, WHFM to RRFM: F = m, alpha <= 0

    elif F == m and alpha <= 0:

        if J <= J_max:

            m_prime = m if m*(d+1) <= J_max else np.inf

            inv_edf = cedf_basic_sum(
                J=J, M=M, S=S, F=m_prime, alpha=alpha, d=d
            ) / (M * cedf_sz(t=0, F=m_prime, alpha=alpha, d=d)**2)

        elif J > J_max and r >= d+1:

            a0, a1 = tables.GREENHALL_TABLE2[alpha][d]

            inv_edf = (a0 - a1/r) / r

        else:

            m_prime = J_max / r  # not necessarily an integer

            inv_edf = cedf_basic_sum(
                J=J_max, M=J_max, S=m_prime, F=np.inf, alpha=alpha, d=d
            ) / (J_max * cedf_sz(t=0, F=np.inf, alpha=alpha, d=d)**2)

    #  3.6.3 Case 3. Unmodified variances, FLPM: F = m, alpha = 1

    elif F == m and alpha == 1:

        if J <= J_max:

            inv_edf = cedf_basic_sum(J=J, M=M, S=S, F=m, alpha=1, d=d) / \
                      (M * cedf_sz(t=0, F=m, alpha=1, d=d)**2)

            if m > 10**6:
                logger.warning("CEDF calculation at this averaging time "
                               "might have a roundoff error.")

        elif J > J_max and r >= d+1:

            a0, a1 = tables.GREENHALL_TABLE2[alpha][d]
            b0, b1 = tables.GREENHALL_TABLE3[alpha][d]

            inv_edf = (a0 - a1/r) / (r * (b0 + b1*np.log(m))**2)

        else:

            m_prime = J_max / r  # not necessarily an integer

            b0, b1 = tables.GREENHALL_TABLE3[alpha][d]

            inv_edf = cedf_basic_sum(
                J=J_max, M=J_max, S=m_prime, F=m_prime, alpha=1, d=d
            ) / (J_max * (b0 + b1*np.log(m))**2)

    #  3.6.4 Case 4. Unmodified variances, WHPM: F = m, alpha = 2

    elif F == m and alpha == 2:

        K = int(np.ceil(r))

        if K <= d:

            s = 0.
            for k in range(1, K):
                s += (1 - k/r) * utils.binom(n=2*d, k=d-k)**2

            s *= (2 / utils.binom(n=2*d, k=d)**2)

            s += 1

            inv_edf = s / M

        else:

            a0, a1 = tables.GREENHALL_TABLE2[alpha][d]

            inv_edf = (a0 - a1/r) / M

    else:

        raise ValueError("Could not calculate equivalent degrees of freedom "
                         "for given combination of input parameters... "
                         "something went wrong!")

    return 1 / inv_edf


def cedf_sw(t: float, alpha: int) -> float:
    """ Calculate the generalised autocovariance (GACV) of the pure
    continuous time power-law process w(t) with spectral density exponent alpha

    References:
        C. Greenhall and W. Riley, "Uncertainty of Stability Variances Based
        on Finite Differences", Proc. 2003 PTTI Meeting , December 2003 (Eq.7)

    Args:
        t:      time at which to compute the metric
        alpha:  frequency power law noise exponent.

    Returns:
        generalised autocovariance at given time for given noise type
    """

    t = np.abs(t)

    if alpha == 2:
        sw = -t

    elif alpha == 1:
        sw = np.log(t) * t**2 if t != 0. else 0

    elif alpha == 0:
        sw = t**3

    elif alpha == -1:
        sw = -np.log(t) * t**4 if t != 0. else 0

    elif alpha == -2:
        sw = -(t**5)

    elif alpha == -3:
        sw = np.log(t) * t**6

    elif alpha == -4:
        sw = t**7

    else:
        raise ValueError(f"Noise type alpha = {alpha} not recognised.")

    return sw


def cedf_sx(t: float, F: float, alpha: int) -> float:
    """ Calculate the phase noise at time t for given dominant noise type
    alpha.

    References:
        C. Greenhall and W. Riley, "Uncertainty of Stability Variances Based
        on Finite Differences", Proc. 2003 PTTI Meeting , December 2003 (Eq.8)

    Args:
        t:      time at which to compute the metric
        F:      filter factor
        alpha:  frequency power law noise exponent.

    Returns:
        phase noise at given time for given noise type
    """

    if np.isinf(F) and (-4 <= alpha <= 0):

        sx = cedf_sw(t=t, alpha=alpha+2)

    else:

        a = 2*cedf_sw(t=t, alpha=alpha)
        b = cedf_sw(t=t-1/F, alpha=alpha)
        c = cedf_sw(t=t+1/F, alpha=alpha)

        sx = (a-b-c) * F**2

    return sx


def cedf_sz(t: float, F: float, alpha: int, d: int) -> float:
    """ Calculate the auto-covariance (ACV) sz at time t.

    References:
        C. Greenhall and W. Riley, "Uncertainty of Stability Variances Based
        on Finite Differences", Proc. 2003 PTTI Meeting , December 2003 (Eq.9)

    Args:
        t:      time at which to compute the metric
        F:      filter factor
        alpha:  frequency power law noise exponent.
        d:      order of phase difference of the variance

    Returns:
        auto-covariance at given time for given noise type
    """

    if d == 1:

        a = cedf_sx(t=t, F=F, alpha=alpha)
        b = cedf_sx(t=t-1, F=F, alpha=alpha)
        c = cedf_sx(t=t+1, F=F, alpha=alpha)

        sz = 2*a - b - c

    elif d == 2:

        a = cedf_sx(t=t, F=F, alpha=alpha)
        b = cedf_sx(t=t-1, F=F, alpha=alpha)
        c = cedf_sx(t=t+1, F=F, alpha=alpha)
        d_ = cedf_sx(t=t-2, F=F, alpha=alpha)
        e = cedf_sx(t=t+2, F=F, alpha=alpha)

        sz = 6*a - 4*b - 4*c + d_ + e

    elif d == 3:

        a = cedf_sx(t=t, F=F, alpha=alpha)
        b = cedf_sx(t=t-1, F=F, alpha=alpha)
        c = cedf_sx(t=t+1, F=F, alpha=alpha)
        d_ = cedf_sx(t=t-2, F=F, alpha=alpha)
        e = cedf_sx(t=t+2, F=F, alpha=alpha)
        f = cedf_sx(t=t-3, F=F, alpha=alpha)
        g = cedf_sx(t=t+3, F=F, alpha=alpha)

        sz = 20*a - 15*b - 15*c + 6*d_ + 6*e - f - g

    else:

        raise ValueError(f"Phase differencing order d = {d} not recognised. "
                         f"Accepted values should be in [1, 2, 3].")

    return sz


def cedf_basic_sum(J: float, M: float, S: float, F: float, alpha: int, d: int):
    """ Calculate the BasicSum factor to obtain the edf for the
    calculated variance estimator under the given dominant noise type alpha.

    References:
        C. Greenhall and W. Riley, "Uncertainty of Stability Variances Based
        on Finite Differences", Proc. 2003 PTTI Meeting , December 2003 (Eq.10)

    Args:
        J:      iteration number?
        M:      number of summands in the estimator
        S:      stride factor
        F:      filter factor
        alpha:  frequency power law noise exponent.
        d:      order of phase difference of the variance

    Returns:
        value of the BasicSum, for edf calculation
    """

    a = cedf_sz(t=0, F=F, alpha=alpha, d=d)**2

    b = (1 - J/M) * cedf_sz(t=J/S, F=F, alpha=alpha, d=d)**2

    c = 0.
    for j in range(1, int(J)):
        c += (1 - j/M) * cedf_sz(t=j/S, F=F, alpha=alpha, d=d)**2
    c *= 2

    basic_sum = a + b + c

    return basic_sum

# -----------------------------------

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
