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
from typing import Tuple


# Spawn module-level logger
logger = logging.getLogger(__name__)

# Confidence Intervals
ONE_SIGMA_CI = 0.683  # scipy.special.erf(1/np.sqrt(2))

# shorten type hint to save some space
Array = np.ndarray


def get_error_bars(x: Array, m: int, var: float, n: int, alpha: int,
                   dev_type: str, ci: float = ONE_SIGMA_CI):
    """Calculate non-naive variance errors. Equivalent to Stable32.

    References:
        [RileyStable32Manual]_ (Confidence Intervals, pg.89)
        http://www.wriley.com/CI2.pdf
        https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/20050061319.pdf

    Args:
        x:          phase data from which deviation was computed, in units of
                    seconds.
        m:          averaging factor at which deviation was computed
        var:        variance value for which to compute error bars
        n:          number of analysis samples used to compute deviation
        alpha:      dominant power law frequency noise type
        dev_type:   type of deviation for which error bars are being
                    calculated, e.g. `adev`.
        ci:         confidence factor for which to set confidence limits.
                    Defaults to 1-sigma confidence intervals.

    Returns:
        lower and upper bounds for variance.
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
    # edf = round(edf, 3)  # this matches stable 32

    # with the known EDF we get CIs
    (lo, hi) = confidence_interval(var=var, edf=edf, ci=ci)

    return lo, hi


def confidence_interval(var: float, edf: float, ci: float) -> \
        Tuple[float, float]:
    """Returns double-sided statistical limits on the true variance at
    requested confidence factor.

    Calculation based on Chi-square statistics of observed sample variance
    and corresponding equivalent degrees of freedom.

    References:
        http://www.wriley.com/CI2.pdf
        https://faculty.elgin.edu/dkernler/statistics/ch09/9-3.html


    Args:
        var:    sample variance from which to calculate confidence intervals
        edf:    equivalent degrees of freedom
        ci:     confidence factor for which to set confidence limits on the
                variance e.g. `0.6827` would set a 1-sigma confidence interval.

    Returns:
        lower and upper bounds for variance.
    """

    chi2_r, chi2_l = scipy.stats.chi2.interval(ci, edf)
    print(f"DF: {edf} \t| X2 = {round(chi2_l, 3), round(chi2_r, 3)}")

    var_lo = edf * var / chi2_l
    var_hi = edf * var / chi2_r

    return var_lo, var_hi

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


def calc_edf_totdev(x: Array, m: int, alpha: int) -> float:
    """Calculate equivalent number of Chi-squared degrees of freedom (edf)
    for TOTDEV.

    References:
        http://www.wriley.com/CI2.pdf (TOTVAR and TTOT EDF)

    Args:
        x:          phase data from which deviation was computed, in units of
                    seconds.
        m:          averaging factor at which deviation was computed
        alpha:      dominant power law frequency noise type

    Returns:
        equivalent number of Chi-squared degrees of freedom (edf)
    """

    try:
        b, c = tables.TOTVAR_EDF_COEFFICIENTS[alpha]

    except KeyError:
        raise ValueError(f"Noise type alpha = {alpha} not supported for "
                         f"(T)TOTDEV edf calculation.")

    edf = b*(x.size/m) - c

    return edf


def calc_edf_ttotdev(x: Array, m: int, alpha: int) -> float:
    """Calculate equivalent number of Chi-squared degrees of freedom (edf)
    for TTOTDEV.

    References:
        http://www.wriley.com/CI2.pdf (TOTVAR and TTOT EDF)

    Args:
        x:          phase data from which deviation was computed, in units of
                    seconds.
        m:          averaging factor at which deviation was computed
        alpha:      dominant power law frequency noise type

    Returns:
        equivalent number of Chi-squared degrees of freedom (edf)
    """

    return calc_edf_totdev(x=x, m=m, alpha=alpha)


def calc_edf_mtotdev(x: Array, m: int, alpha: int) -> float:
    """Calculate equivalent number of Chi-squared degrees of freedom (edf)
    for MTOTDEV.

    References:
        http://www.wriley.com/CI2.pdf (MTOT EDF)

    Args:
        x:          phase data from which deviation was computed, in units of
                    seconds.
        m:          averaging factor at which deviation was computed
        alpha:      dominant power law frequency noise type

    Returns:
        equivalent number of Chi-squared degrees of freedom (edf)
    """

    try:

        b, c = tables.MTOTVAR_EDF_COEFFICIENTS[alpha]

    except KeyError:
        raise ValueError(f"Noise type alpha = {alpha} not supported for "
                         f"MTOT edf calculation.")

    edf = b*(x.size/m) - c

    return edf


def calc_edf_htotdev(x: Array, m: int, alpha: int) -> float:
    """Calculate equivalent number of Chi-squared degrees of freedom (edf)
    for HTOTDEV.

    References:
        D.A. Howe, et. Al., “A Total Estimator of the Hadamard Function Used
        for GPS Operations”, Proc. 32nd PTTI Meeting, pp. 255-268, November
        2000

        http://www.wriley.com/CI2.pdf (HTOT EDF)

    Args:
        x:          phase data from which deviation was computed, in units of
                    seconds.
        m:          averaging factor at which deviation was computed
        alpha:      dominant power law frequency noise type

    Returns:
        equivalent number of Chi-squared degrees of freedom (edf)
    """

    if m < 16:  # see Stable32 Help Manual and paper

        # Use plain Hadamard estimator
        return calc_edf_hdev(x=x, m=m, alpha=alpha)

    try:
        b0, b1 = tables.HTOTVAR_EDF_COEFFICIENTS[alpha]

    except KeyError:
        raise ValueError(f"Noise type alpha = {alpha} not supported for "
                         f"HTOTDEV edf calculation.")

    edf = (x.size/m) / (b0 + b1*m/x.size)

    return edf


def calc_edf_theo1(x: Array, m: int, alpha: int) -> float:
    """Calculate equivalent number of Chi-squared degrees of freedom (edf)
    for THEO1.

    References:

        D.A. Howe and T.K. Peppler, “Estimation of Very Long-Term Frequency
        Stability Using a Special-Purpose Statistic”, ", Proc. 2003 Joint
        Meeting of the European Freq. and Time Forum and the IEEE
        International Freq. Contrl. Symp., May 2003


        http://www.wriley.com/CI2.pdf (THEO1 EDF)

    Args:
        x:          phase data from which deviation was computed, in units of
                    seconds.
        m:          averaging factor at which deviation was computed
        alpha:      dominant power law frequency noise type

    Returns:
        equivalent number of Chi-squared degrees of freedom (edf)
    """

    N = x.size
    t = 0.75*m  # effective theo1 averaging factor

    if alpha == 2:

        edf = ((0.86 * (N+1) * (N - 4*t/3)) / (N - t)) * (t / (t + 1.14))

    elif alpha == 1:

        edf = ((4.798*N**2 - 6.374*N*t + 12.387*t) / (np.sqrt(t+36.6) * (
                N-t))) * (t / (t + 0.3))

    elif alpha == 0:

        edf = ((4.1*N+0.8)/t - (3.1*N+6.5)/N) * (t**1.5 / (t**1.5 + 5.2))

    elif alpha == -1:

        edf = ((2*N**2 - 1.3*N*t - 3.5*t) / (N*t)) * (t**3 / (t**3 + 2.3))

    elif alpha == -2:

        edf = ((4.4*N-2) / (2.9*t)) * (
                ((4.4*N-1)**2 - 8.6*t*(4.4*N-1) + 11.4*t**2) / (4.4*N-3)**2)

    else:
        raise ValueError(f"Noise type alpha = {alpha} not supported for "
                         f"Theo1 edf calculation.")

    return edf


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


# TODO: check and see if should use this when other edf functions don't
#  support the given noise type
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
